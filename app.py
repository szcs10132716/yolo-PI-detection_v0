# app.py

# Python In-built packages
from pathlib import Path
from PIL import Image
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv8 (with ArUco measurement)")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = Image.open(source_img)
                st.image(uploaded_image, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            # Show default detection result image
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                # Convert PIL image to numpy array (OpenCV format)
                image_array = np.array(uploaded_image.convert('RGB'))
                # Perform detection + measurement
                annotated_img, detections_info = helper.detect_and_measure(
                    image_array, model, conf=confidence
                )
                st.image(annotated_img, caption='Detected & Measured Image',
                         use_column_width=True)

                # 显示详细的检测结果信息 # 显示检测到的宽高等信息
                with st.expander("Detection Results"):
                    for info in detections_info:
                        st.write(info)

elif source_radio == settings.WEBCAM:
    # Webcam
    st.sidebar.write("Please use the camera below to take a picture or upload an image.")
    picture = st.camera_input("Take a picture or upload an image")

    if picture:
        img = Image.open(picture)
        st.image(img, caption='Input Image', use_column_width=True)

        if st.sidebar.button('Detect Objects'):
            image_array = np.array(img.convert('RGB'))
            annotated_img, detections_info = helper.detect_and_measure(
                image_array, model, conf=confidence
            )
            st.image(annotated_img, caption='Detected & Measured Image', use_column_width=True)

            with st.expander("Detection Results"):
                for info in detections_info:
                    st.write(info)

elif source_radio == settings.VIDEO:
    # Play a stored video file and perform detection + measurement
    helper.play_stored_video(confidence, model)

elif source_radio == settings.RTSP:
    # Play RTSP stream
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    # Play YouTube video
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
