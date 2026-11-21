import hashlib
import io
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import PIL
import streamlit as st
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "DIP" / "training" / "src"))
sys.path.insert(0, str(REPO_ROOT / "CN" / "src"))

EDGE_MODEL = (
    REPO_ROOT / "model_share/deploy/model_int8.onnx"
)
EDGE_LABELS = (
    REPO_ROOT / "model_share/deploy/labels.json"
)
EDGE_PREPROCESSING = (
    REPO_ROOT / "model_share/deploy/preprocessing.json"
)
POLICY_PATH = (
    REPO_ROOT / "CN/results/cn_edge_fog/policy_tree.joblib"
)


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


EDGE_MODEL_SHA256 = _sha256(EDGE_MODEL)
GIT_COMMIT = _git_commit()
MANIFEST_PATH = REPO_ROOT / "DIP/training/data/manifests/quality_features.csv"


@st.cache_data(show_spinner=False)
def load_demo_reference() -> dict[str, tuple[str, str]]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        frame = pd.read_csv(
            MANIFEST_PATH,
            usecols=["image_id", "class_label", "severity_label"],
            dtype={"image_id": str},
        )
    except Exception:
        return {}
    return {
        str(row.image_id).zfill(8): (
            str(row.class_label),
            str(row.severity_label),
        )
        for row in frame.itertuples(index=False)
    }


FOG_CHECKPOINTS = [
    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed_focal_reran/best.pt",

    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed_eff/best.pt",

    REPO_ROOT / "DIP/training/checkpoints/"
    "effb0_224_bs64_mixed/best.pt",
]

# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WBC Classification | Edge–Fog AI",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stHeader"] { background: #0d151c; color: #e6f1f4; }
.stApp h1, .stApp h2, .stApp h3 { color: #e6f1f4; }
.stApp h1 { font-size: clamp(1.65rem, 4vw, 2.5rem); }
[data-testid="stMetric"] { background: #16252e; border: 1px solid #29434c; border-radius: .6rem; padding: .75rem; }
[data-testid="stMetricValue"] { color: #75e3d2; font-size: clamp(1.1rem, 3vw, 1.8rem); }
[data-testid="stBaseButton-primary"] { background: #75e3d2; border-color: #75e3d2; color: #0d151c; }
[data-testid="stBaseButton-primary"]:hover { background: #abf6e9; color: #0d151c; }
[role="progressbar"] > div > div { background-color: #52cbb8; }
</style>
""", unsafe_allow_html=True)
st.title("WBC Classification | Edge–Fog AI")

st.write(
    "13-class white blood cell classification using image quality "
    "assessment and adaptive Edge–Fog routing."
)

st.info(
    "Research prototype: predictions are experimental and "
    "must not be used as a clinical diagnosis."
)

# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

try:
    from cn.edge.local_predictor import OnnxEdgePredictor
    from cn.edge.policy import LearnedPolicy
    from cn.edge.profiles import get
    from cn.client import FogClient
    from cn.server import FogServer
    from cn.fog_backend import TorchFogBackend

    from quality.degradation import describe_quality
    from quality.features import extract_image_features

except Exception as e:
    st.error(f"Unable to import project modules: {e}")
    st.stop()


# --------------------------------------------------
# LOAD EDGE, POLICY AND FOG SERVER ONCE
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading Edge and Fog models...")
def load_pipeline():

    required_paths = [
        EDGE_MODEL,
        EDGE_LABELS,
        EDGE_PREPROCESSING,
        POLICY_PATH,
        *FOG_CHECKPOINTS
    ]

    missing = [str(p) for p in required_paths if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Required model files are missing:\n" + "\n".join(missing)
        )

    # Load INT8 Edge model
    edge = OnnxEdgePredictor(
        model_path=str(EDGE_MODEL),
        labels_path=str(EDGE_LABELS),
        preprocessing_path=str(EDGE_PREPROCESSING),
        predict_script=str(
            REPO_ROOT / "model_share/deploy/predict.py"
        ),
        threads=1
    )

    # Load saved learned routing policy
    policy = LearnedPolicy.load(str(POLICY_PATH))

    # Load three-checkpoint Fog ensemble
    fog_backend = TorchFogBackend(
        [str(p) for p in FOG_CHECKPOINTS],
        device="cpu",
        tta="full",
        model_name="fog_ensemble",
        num_threads=2,
        warmup=False
    )

    # Start real TCP Fog server
    server = FogServer(
        "127.0.0.1",
        0,
        backend=fog_backend,
        timeout_s=180.0
    )

    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )

    server_thread.start()

    host, port = server.server_address

    # Create TCP client
    fog_client = FogClient(
        host=host,
        port=port,
        timeout_s=180.0,
        connect_timeout_s=10.0
    )

    return {
        "edge": edge,
        "policy": policy,
        "fog_client": fog_client,
        "server": server,
        "server_thread": server_thread,
        "fog_lock": threading.Lock(),
        "fog_backend": fog_backend
    }




PROFILES = {"WiFi": "wifi", "5G": "5g", "Poor WiFi": "poor_wifi", "LPWAN": "lpwan"}
FEATURES = {
    "laplacian_variance": ("Laplacian variance", "Variance of the grayscale Laplacian."),
    "median_filter_residual": ("Median-filter residual", "Mean absolute change after a 3 × 3 median filter."),
    "brightness": ("Brightness", "Mean grayscale intensity."),
    "contrast": ("Contrast", "Standard deviation of grayscale intensity."),
    "extreme_pixel_percentage": ("Extreme pixels (%)", "Percentage of grayscale pixels at or below 5 or at or above 250."),
    "rgb_colour_variation": ("RGB colour variation", "Mean of the three colour-channel standard deviations."),
    "edge_density": ("Edge density (%)", "Percentage of pixels detected as edges by Canny at thresholds 100 and 200."),
}


def section(title):
    st.divider()
    st.subheader(title)


def milliseconds(value):
    return f"{value:,.2f} ms"


def show_top3(items):
    st.markdown("**Top-3 predictions**")
    for item in items:
        if isinstance(item, dict):
            label = item.get("label", item.get("class", item.get("predicted_class", "Unknown")))
            score = item.get("confidence", item.get("probability", item.get("score", 0)))
        else:
            label, score = item[:2]
        st.progress(float(score), text=f"{label} · {float(score):.2%}")


def classify_on_fog(pipeline, path, image_name):
    with st.spinner("Sending the image to Fog over localhost TCP..."):
        with pipeline["fog_lock"]:
            result = pipeline["fog_client"].classify(path, image_id=image_name)
        if not result.get("integrity_ok", False):
            raise RuntimeError("Image integrity verification failed.")
    return result


def run_analysis(pipeline, image_bytes, image_name, profile_name, suffix):
    started = perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / f"image{suffix}")
        Path(path).write_bytes(image_bytes)
        with st.spinner("Extracting image quality features..."):
            features = extract_image_features(path)
            quality = describe_quality(features)
        with st.spinner("Running the Edge INT8 model..."):
            edge = pipeline["edge"].predict(path)
            decision = pipeline["policy"].decide(features, edge, get(profile_name))

        action = decision.value.lower()
        fog = None
        fog_error = None
        if action == "offload":
            try:
                fog = classify_on_fog(pipeline, path, image_name)
            except Exception as error:
                fog_error = str(error)
        elif action != "local":
            raise RuntimeError(f"Unknown routing decision: {action}")

        pipeline_ms = (perf_counter() - started) * 1000.0

    return {
        "features": features,
        "quality": quality,
        "edge": edge,
        "action": action,
        "fog": fog,
        "fog_error": fog_error,
        "pipeline_ms": pipeline_ms,
    }


def show_fog_comparison(analysis, image_bytes, image_name, suffix):
    if st.button("Run Fog prediction"):
        analysis.pop("manual_fog", None)
        try:
            pipeline = load_pipeline()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"image{suffix}"
                path.write_bytes(image_bytes)
                analysis["manual_fog"] = classify_on_fog(pipeline, str(path), image_name)
        except Exception as error:
            st.error(f"On-demand Fog prediction failed. {error}")

    fog = analysis.get("manual_fog")
    if fog is not None:
        with st.container(border=True):
            st.markdown("**On-demand Fog prediction**")
            fog_class, confidence, exchange = st.columns(3)
            fog_class.metric("Fog class", fog["predicted_class"])
            confidence.metric("Fog confidence", f"{fog['confidence']:.2%}")
            exchange.metric("TCP/Fog exchange", milliseconds(fog["total_ms"]))
            show_top3(fog["top3"])


def show_analysis(analysis, selected_profile, image_bytes, image_name, suffix):
    section("3. Image Quality Assessment")
    rows = [
        {"Quality feature": label, "Value": analysis["features"][key]}
        for key, (label, _) in FEATURES.items()
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Value": st.column_config.NumberColumn(format="%.6f")},
    )
    with st.expander("Feature definitions"):
        for label, definition in FEATURES.values():
            st.markdown(f"**{label}** · {definition}")
    quality = analysis["quality"]
    appearance, severity = st.columns(2)
    appearance.metric("Estimated appearance", quality["appearance"])
    severity.metric("Estimated severity band", quality["severity"])
    st.caption(f"{quality['summary']}. Estimated from image-quality features.")
    reference = load_demo_reference().get(Path(image_name).stem.zfill(8))
    if reference is not None:
        demo_class, demo_severity = reference
        st.info(
            f"Filename match in dataset manifest: class **{demo_class}**, "
            f"dataset severity **{demo_severity}**."
        )

    edge = analysis["edge"]
    section("4. Learned Edge/Fog Routing")
    local = analysis["action"] == "local"
    source = "Edge INT8" if local else "Fog ensemble"
    route = "LOCAL" if local else "OFFLOAD"
    st.info(f"**{route} · {source}** · {selected_profile}")
    if local:
        st.caption("Image quality → Edge INT8 → Learned policy → Edge result")
    else:
        st.caption("Image quality → Edge INT8 → Learned policy → Localhost TCP → Fog ensemble")

    section("5. Final WBC Classification Result")
    st.info("Research use only. This experimental prediction is not a medical diagnosis.")
    fog = analysis["fog"]
    final = edge if local else fog
    if final is None:
        st.error(f"Fog classification failed on localhost TCP (127.0.0.1). {analysis['fog_error']}")
    else:
        with st.container(border=True):
            final_class, confidence, prediction_source = st.columns(3)
            final_class.metric("WBC class", final["predicted_class"])
            confidence.metric("Confidence", f"{final['confidence']:.2%}")
            prediction_source.metric("Prediction source", source)
            show_top3(final["top3"])
        st.caption(
            "Confidence is the model’s score for the selected class. "
            "Lower scores indicate more uncertainty."
        )

    if local:
        show_fog_comparison(analysis, image_bytes, image_name, suffix)

    with st.expander("Edge prediction used for routing"):
        st.caption("The learned policy uses the Edge INT8 prediction for every image.")
        edge_class, edge_score, edge_time = st.columns(3)
        edge_class.metric("Edge class", edge["predicted_class"])
        edge_score.metric("Edge confidence", f"{edge['confidence']:.2%}")
        edge_time.metric("Edge inference", milliseconds(edge["latency_ms"]))
        show_top3(edge["top3"])

    section("6. Performance and Communication")
    st.caption(f"Predefined network profile · {selected_profile}")
    edge_timing, fog_timing, pipeline_timing = st.columns(3)
    edge_timing.metric("Edge inference", milliseconds(edge["latency_ms"]))
    if fog is not None:
        fog_timing.metric("TCP/Fog exchange", milliseconds(fog["total_ms"]))
    if final is not None:
        pipeline_timing.metric("Analysis pipeline", milliseconds(analysis["pipeline_ms"]))
        st.caption(
            "Pipeline time is measured from preparing the validated upload for analysis "
            "through the final result. It includes quality extraction, Edge inference, "
            "routing, and any Fog call. Model startup, upload to Streamlit, "
            "image validation, and page rendering are outside this timer."
        )

    if local:
        st.info("The learned policy completed this analysis on Edge.")
    elif fog is not None:
        st.markdown("**TCP communication · Localhost (127.0.0.1)**")
        upload, inference, download = st.columns(3)
        upload.metric("Upload", milliseconds(fog["upload_ms"]))
        inference.metric("Fog inference", milliseconds(fog["latency_ms"]))
        download.metric("Download (derived)", milliseconds(fog["download_ms"]))
        st.caption(
            "The TCP client times the exchange from the file metadata request through "
            "the verified result. Download is calculated from the exchange time after "
            "subtracting upload and server processing."
        )
        sent, received, integrity = st.columns(3)
        sent.metric("Bytes sent", f"{fog['bytes_sent']:,d}")
        received.metric("Bytes received", f"{fog['bytes_received']:,d}")
        integrity.metric("TCP image integrity", "Verified")


section("1. Network Configuration")
selected_profile = st.selectbox(
    "Network profile",
    list(PROFILES),
    help="Predefined inputs to the learned routing policy. These profiles do not measure your connection.",
)
profile_name = PROFILES[selected_profile]

section("2. Upload WBC Image")
uploaded_file = st.file_uploader("Choose a WBC image", type=["jpg", "jpeg", "png"])
st.session_state.setdefault("wbc_analysis", None)

if uploaded_file is None:
    st.session_state.wbc_analysis = None
    st.info("Upload a JPG or PNG image to continue.")
else:
    image_bytes = uploaded_file.getvalue()
    image_name = uploaded_file.name
    input_key = (hashlib.sha256(image_bytes).hexdigest(), image_name, profile_name)
    saved = st.session_state.wbc_analysis
    if saved is not None and saved["key"] != input_key:
        st.session_state.wbc_analysis = None

    try:
        with Image.open(io.BytesIO(image_bytes)) as check:
            image_format = check.format
            if image_format not in {"JPEG", "PNG"}:
                raise ValueError("Choose a JPEG or PNG image.")
            if min(check.size) < 10:
                raise ValueError("The image must be at least 10 × 10 pixels.")
            check.verify()
        with Image.open(io.BytesIO(image_bytes)) as original:
            image = original.convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        st.session_state.wbc_analysis = None
        st.error(f"Unable to read the uploaded image. {error}")
    else:
        suffix = ".png" if image_format == "PNG" else ".jpg"
        preview = image.copy()
        preview.thumbnail((300, 240))
        with st.container(border=True):
            preview_column, file_column = st.columns(2)
            preview_column.image(preview, caption="Uploaded WBC image")
            file_column.text(image_name)
            file_column.write(f"{image.width} × {image.height} px · {image_format}")
            file_column.caption(f"RGB preview · {len(image_bytes):,d} bytes")
        if st.button("Analyze Image", type="primary"):
            st.session_state.wbc_analysis = None
            try:
                pipeline = load_pipeline()
            except Exception as error:
                st.error(f"Unable to initialize the Edge and Fog models or the TCP server. {error}")
            else:
                try:
                    result = run_analysis(pipeline, image_bytes, image_name, profile_name, suffix)
                except Exception as error:
                    st.error(f"Image analysis failed. {error}")
                else:
                    st.session_state.wbc_analysis = {"key": input_key, "result": result}

        if st.session_state.wbc_analysis is not None:
            show_analysis(
                st.session_state.wbc_analysis["result"], selected_profile,
                image_bytes, image_name, suffix,
            )

st.divider()
st.caption(
    f"Build `{GIT_COMMIT}` | Edge model SHA-256 `{EDGE_MODEL_SHA256}` | "
    f"onnxruntime {ort.__version__} | Pillow {PIL.__version__}"
)
