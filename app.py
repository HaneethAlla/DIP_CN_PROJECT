
import streamlit as st
from PIL import Image
import io

st.set_page_config(
    page_title="WBC Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 WBC Cancer Detection System")
st.write(
    "Upload a White Blood Cell image to begin quality assessment "
    "and segmentation."
)

st.divider()

st.subheader("1. Upload WBC Image")

uploaded_file = st.file_uploader(
    "Choose a WBC image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image in JPG, JPEG or PNG format."
)

if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if image.width < 10 or image.height < 10:
            st.error("Image dimensions are too small.")
        else:
            st.success("Image uploaded successfully!")

            st.subheader("2. Uploaded Image")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Uploaded WBC Image",
                    width="stretch"
                )

            with col2:
                st.write("### Image Information")
                st.write(f"**File name:** {uploaded_file.name}")
                st.write(f"**Image format:** {Image.open(io.BytesIO(image_bytes)).format}")
                st.write(f"**Width:** {image.width} pixels")
                st.write(f"**Height:** {image.height} pixels")
                st.write(f"**Color channels:** 3 (RGB)")
                st.write(f"**File size:** {len(image_bytes) / 1024:.2f} KB")

            st.divider()

            st.subheader("3. Image Analysis")

            st.info(
                "Image upload is complete. Quality feature extraction, "
                "preprocessing and Cellpose segmentation will be connected "
                "in the next integration stage."
            )

            if st.button("Analyze Image", type="primary"):
                st.warning(
                    "Image processing is not connected yet. "
                    "This feature will be enabled after integration."
                )

    except Exception:
        st.error(
            "Unable to read this image. Please upload a valid JPG, JPEG or PNG file."
        )

else:
    st.info("Please upload a WBC image to continue.")
