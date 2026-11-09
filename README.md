\# WBC Cancer Detection System



A web-based frontend for uploading and previewing White Blood Cell (WBC) images as part of a WBC cancer detection project.



\## Current Features



\- Upload WBC images in JPG, JPEG and PNG formats.

\- Display the uploaded image.

\- Show image information:

&#x20; - File name

&#x20; - Image format

&#x20; - Width and height

&#x20; - Colour channels

&#x20; - File size

\- Validate uploaded image files.



\## Technologies Used



\- Python

\- Streamlit

\- Pillow



\## Installation



1\. Install Python.

2\. Open a terminal in the project folder.

3\. Install the required packages:



```bash

pip install -r requirements.txt

```



\## Run the Application



```bash

streamlit run app.py

```



Open the local URL displayed in the terminal, usually:



`http://localhost:8501`



\## Current Development Status



The image upload and preview interface is implemented and tested.



Image quality feature extraction, preprocessing, Cellpose segmentation and cancer classification are not yet integrated.



\## Project Scope



This project focuses on cancer prediction using 13 types of White Blood Cells.

