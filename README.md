代码A：

```
import os
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 获取当前脚本所在目录，或者直接用 os.getcwd()
    current_dir = os.getcwd()
    # 假设 best.pt 跟脚本同级
    # model_path = os.path.join(current_dir, "best.pt")

    # 如果你确实有个 checkpoint 文件夹，而且 best.pt 放在里边
    model_path = os.path.join(current_dir, "checkpoint", "best_v8l_PI.pt")
    
    # 打印一下模型路径，确认是否真的存在
    print("正在加载模型：", model_path)
    if not os.path.exists(model_path):
        print("错误：模型文件不存在，请检查路径。")
        return

    # 加载YOLOv8模型
    model = YOLO(model_path)

    # 加载 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 初始化摄像头
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头。请检查摄像头连接和索引（{camera_index}）。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pixel_cm_ratio = None

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            print("无法读取摄像头图像，请检查连接。")
            break

        # 检测 ArUco 标记
        corners, ids, rejected = detector.detectMarkers(img)
        if ids is not None and len(ids) > 0:
            aruco_perimeter = cv2.arcLength(corners[0][0], True)
            pixel_cm_ratio = aruco_perimeter / 20

            # 绘制 ArUco 标记
            cv2.polylines(img, [corners[0][0].astype(int)], True, (0, 255, 0), 5)
            cv2.putText(img,
                        f"Aruco ID: {ids[0][0]}",
                        (int(corners[0][0][0][0]), int(corners[0][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 使用 YOLO 进行目标检测
        results = model(img)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # 绘制边界框和标签
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{class_name} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 计算并标注尺寸
                if pixel_cm_ratio:
                    width_cm = (x2 - x1) / pixel_cm_ratio
                    height_cm = (y2 - y1) / pixel_cm_ratio
                    cv2.putText(img, f"W:{width_cm:.1f}cm H:{height_cm:.1f}cm",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("压力性损伤检测与测量", img)
        key = cv2.waitKey(1)
        if key == 27:  # 按下Esc键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```





“代码A”是我的代码，能够顺利运行，执行的任务是 YOLO目标检测+ArUco尺寸测量。下面是我的一个项目，包括这个项目的目录树，以及部分项目文件的内容；我想在这个项目的基础上进行修改，执行目标检测的时候，能够同时测量检测目标的尺寸（参考上面的“代码A”）。


目录树

yolo-PI-detection_v0/
├── .devcontainer/
│   └── devcontainer.json
├── assets/
│   ├── Objdetectionyoutubegif-1.m4v
│   ├── pic1.png
│   ├── pic3.png
│   └── segmentation.png
├── images/
│   ├── office_4.jpg
│   └── office_4_detected.jpg
├── videos/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── video_3.mp4
├── weights/
│   ├── yolov8n-cls.pt
│   ├── yolov8n-seg.pt
│   └── yolov8n.pt
├── .gitattributes
├── .gitignore
├── README.md
├── app.py
├── helper.py

├── packages.txt
├── requirements.txt

└── settings.py

### 


app.py:
```python
# app.py

# Python In-built packages
from pathlib import Path
from PIL import Image

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
st.title("Object Detection And Tracking using YOLOv8")

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
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    st.sidebar.write("Please use the camera below to take a picture or upload an image.")
    picture = st.camera_input("Take a picture or upload an image")

    if picture:
        # Read image
        img = Image.open(picture)
        st.image(img, caption='Input Image', use_column_width=True)
        # Perform object detection
        if st.sidebar.button('Detect Objects'):
            res = model.predict(img, conf=confidence)
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
            # Display detection results
            with st.expander("Detection Results"):
                for box in res[0].boxes:
                    st.write(box.data)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

```

helper.py:
```python
# helper.py

import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def load_model(model_path):
    model = YOLO(model_path)
    return model

def play_stored_video(confidence, model):
    # Function to play a stored video file and perform detection
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
            res = model.predict(frame, conf=confidence)
            res_plotted = res[0].plot()
            stframe.image(res_plotted, channels="BGR")
        cap.release()

def play_webcam(confidence, model):
    # Function to access webcam and perform detection
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model
            self.confidence = confidence

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            res = self.model.predict(img, conf=self.confidence)
            res_plotted = res[0].plot()
            return res_plotted

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

def play_rtsp_stream(confidence, model):
    # Function to play RTSP stream and perform detection
    rtsp_url = st.sidebar.text_input("Enter RTSP URL")
    if rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = model.predict(frame, conf=confidence)
            res_plotted = res[0].plot()
            stframe.image(res_plotted, channels="BGR")
        cap.release()

def play_youtube_video(confidence, model):
    # Function to play YouTube video and perform detection
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url:
        st.write("YouTube video playback is not implemented.")

```

packages.txt:

```
freeglut3-dev
libgtk2.0-dev
libgl1-mesa-glx
```

requirements.txt:

```
altair==5.3.0
attrs==23.2.0
blinker==1.8.2
Brotli==1.1.0
cachetools==5.4.0
certifi==2024.7.4
charset-normalizer==3.3.2
click==8.1.7
contourpy==1.2.1
cycler==0.12.1
filelock==3.15.4
fonttools==4.53.1
fsspec==2024.6.1
gitdb==4.0.11
GitPython==3.1.43
idna==3.7
Jinja2==3.1.4
jsonschema==4.23.0
jsonschema-specifications==2023.12.1
kiwisolver==1.4.5
lapx==0.5.9.post1
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.1
mdurl==0.1.2
mpmath==1.3.0
mutagen==1.47.0
networkx==3.3
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.5.82
nvidia-nvtx-cu12==12.1.105
opencv-python==4.10.0.84
packaging==24.1
pafy==0.5.5
pandas==2.2.2
pillow==10.4.0
protobuf==5.27.2
psutil==6.0.0
py-cpuinfo==9.0.0
pyarrow==17.0.0
pycryptodomex==3.20.0
pydeck==0.9.1
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
referencing==0.35.1
requests==2.32.3
rich==13.7.1
rpds-py==0.19.0
scipy==1.14.0
seaborn==0.13.2
six==1.16.0
smmap==5.0.1
streamlit==1.36.0
sympy==1.13.0
tenacity==8.5.0
toml==0.10.2
toolz==0.12.1
torch==2.3.1
torchvision==0.18.1
tornado==6.4.1
tqdm==4.66.4
triton==2.3.1
typing_extensions==4.12.2
tzdata==2024.1
ultralytics==8.2.60
ultralytics-thop==2.0.0
urllib3==2.2.2
watchdog==4.0.1
websockets==12.0
yt-dlp==2024.7.16
av
streamlit-webrtc
```


settings.py:
```python
# settings.py


from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())



# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]


# Paths to the pre-trained models
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'


# Default images
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'
```

