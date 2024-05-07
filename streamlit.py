import streamlit as st
import cv2
import os
from utils.face_detection import FaceDetection


st.title("Face Detection with YOLOv8")

img_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


for n, img_file_buffer in enumerate(img_files):
  if img_file_buffer is not None:
    detection = FaceDetection(img_file_buffer)
    detect_face = detection.detect_face()
    
    if detect_face is not None:
      st.image(detect_face, channels="BGR", \
      caption=f'Detection Results ({n+1}/{len(img_files)})')
        
