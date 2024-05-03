import streamlit as st
import cv2
import os
from utils.face_detection import FaceDetection

UPLOAD_DEST = "./static/uploads"
RESULTS_DEST = "./static/results"

st.title("Face Detection with YOLOv8")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DEST, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make instance of FaceDetection class
    face_detection = FaceDetection(file_path)
    # Detect face
    result = face_detection.detect_face()
    # Visualize the result
    annotated_image = result.plot()
    # Save the annotated image
    result_name = uploaded_file.name.split('.')[0] + '_anotated.jpg'
    cv2.imwrite(os.path.join(RESULTS_DEST, result_name), annotated_image)
    
    result_image_path = os.path.join(RESULTS_DEST, result_name)
    st.image(result_image_path, caption="Annotated Image", use_column_width=True)

