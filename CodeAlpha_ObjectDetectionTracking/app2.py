import streamlit as st
import cv2
import tempfile
import os

from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Object Detection and Tracking")

st.write(
    "Upload a video to detect and track objects "
    "using YOLO and ByteTrack."
)

st.divider()

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# Upload video
uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Start Object Detection"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_input:

            temp_input.write(uploaded_file.read())
            input_path = temp_input.name

        output_path = os.path.join(
            tempfile.gettempdir(),
            "detected_output.mp4"
        )

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        progress = st.progress(0)

        status = st.empty()

        frame_count = 0

        try:

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                # YOLO object detection and ByteTrack tracking
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=0.25,
                    verbose=False
                )

                # Draw bounding boxes and tracking IDs
                annotated_frame = results[0].plot()

                writer.write(annotated_frame)

                frame_count += 1

                if total_frames > 0:
                    progress.progress(
                        min(frame_count / total_frames, 1.0)
                    )

                status.write(
                    f"Processed frames: {frame_count}/{total_frames}"
                )

        finally:
            cap.release()
            writer.release()
            os.remove(input_path)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:

            st.success("Object detection completed!")

            st.subheader("Detection Results")

            st.video(output_path)

            with open(output_path, "rb") as file:

                st.download_button(
                    "Download Processed Video",
                    data=file,
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )

        else:
            st.error("Video processing failed.")

st.divider()

st.caption(
    "CodeAlpha AI Internship | Object Detection and Tracking"
)