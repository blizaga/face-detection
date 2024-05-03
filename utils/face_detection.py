from ultralytics import YOLO
import cv2
import os

FACE_DETECTION_PATH = os.path.join("models", "yolov8n-face.pt")

# Make class for face detection with image and save the inference image
class FaceDetection:
    def __init__(self, image_path, device="cpu"):
        """
        Initialize the FaceDetection class.

        Args:
            image_path (str): The path to the image file.
            model_path (str): The path to the YOLOv8 model file.
            device (str, optional): The device to run the model on. Defaults to "cpu".
        """
        self.image_path = image_path
        self.device = device

    def detect_face(self):
        """
        Detect faces in the image using YOLOv8 model.

        Returns:
            list: A list of bounding boxes representing the detected faces.
        """
        # Load the YOLOv8 model
        model = YOLO(FACE_DETECTION_PATH)

        # Read the image
        image = cv2.imread(self.image_path)

        # Run YOLOv8 on the image
        result = model.predict(image, device=self.device, verbose=True)[0]

        return result

