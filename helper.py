# helper.py

import cv2
import numpy as np
import tempfile
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def load_model(model_path):
    model = YOLO(model_path)
    return model

def detect_and_measure(img, model, conf=0.4):
    """
    Detect objects in the given image using YOLO, and measure each object's 
    width and height in centimeters if an ArUco marker is found.
    
    # 参数：
    #   img: 输入图像（numpy数组，BGR格式或RGB格式都可，但需统一处理）
    #   model: YOLO模型实例
    #   conf: 置信度阈值
    
    # 返回：
    #   annotated_img: 标注后的图像（numpy数组，RGB格式）
    #   detections_info: 检测结果信息的列表（包括类别、置信度、尺寸等）
    """
    # Convert to BGR if needed
    # Some images (from PIL) might be in RGB, so we ensure we work in BGR for OpenCV operations.
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Here we assume it's RGB, convert it to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # Already in BGR
        img_bgr = img

    # Initialize ArUco detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)  # 使用5X5_100字典 # 可以根据需要修改字典
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    pixel_cm_ratio = None

    # Detect ArUco marker
    corners, ids, rejected = detector.detectMarkers(img_bgr)
    if ids is not None and len(ids) > 0:
        # 计算该ArUco的周长(取第一个检测到的marker)
        aruco_perimeter = cv2.arcLength(corners[0][0], True)
        # 假设Aruco的边长为5cm，这里计算周长是 4 x 5 = 20cm
        # 也可以依据实际marker大小进行修改
        pixel_cm_ratio = aruco_perimeter / 20
        
        # 绘制 ArUco 标记
        cv2.polylines(img_bgr, [corners[0][0].astype(int)], True, (0, 255, 0), 5)
        cv2.putText(img_bgr,
                    f"Aruco ID: {ids[0][0]}",
                    (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Perform YOLO detection
    results = model.predict(img_bgr, conf=conf)

    detections_info = []
    annotated_img = img_bgr.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw label
            cv2.putText(annotated_img, f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Compute size if pixel_cm_ratio is available
            width_cm = None
            height_cm = None
            if pixel_cm_ratio:
                width_cm = (x2 - x1) / pixel_cm_ratio
                height_cm = (y2 - y1) / pixel_cm_ratio
                cv2.putText(annotated_img, f"W:{width_cm:.1f}cm H:{height_cm:.1f}cm",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Save detection info
            detection_dict = {
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "bounding_box": (x1, y1, x2, y2),
                "width_cm": round(width_cm, 2) if width_cm else None,
                "height_cm": round(height_cm, 2) if height_cm else None
            }
            detections_info.append(detection_dict)

    # Convert annotated image back to RGB for Streamlit display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return annotated_img, detections_info

def play_stored_video(confidence, model):
    """
    Play a stored video file from user upload and perform YOLO detection + ArUco measurement.
    
    # 参数：
    #   confidence: 置信度阈值
    #   model: YOLO模型实例
    """
    video_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "mov", "avi", "mkv"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # detect_and_measure要求输入BGR格式，因此这里可直接传frame
            annotated_frame, _ = detect_and_measure(frame, model, conf=confidence)
            stframe.image(annotated_frame, channels="RGB")
        cap.release()

def play_webcam(confidence, model):
    """
    Access webcam using streamlit-webrtc and perform detection + measurement.
    """

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.confidence = confidence

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_img, _ = detect_and_measure(img, self.model, conf=self.confidence)
            return annotated_img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

def play_rtsp_stream(confidence, model):
    """
    Play RTSP stream and perform YOLO detection + ArUco measurement.
    """
    rtsp_url = st.sidebar.text_input("Enter RTSP URL")
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, _ = detect_and_measure(frame, model, conf=confidence)
            stframe.image(annotated_frame, channels="RGB")
        cap.release()

def play_youtube_video(confidence, model):
    """
    Play YouTube video and perform YOLO detection + ArUco measurement.
    (Not fully implemented here)
    """
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url:
        st.write("YouTube video playback is not implemented.")
