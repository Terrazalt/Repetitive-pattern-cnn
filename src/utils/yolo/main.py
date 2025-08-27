import torch
from ultralytics import YOLO as _YOLO
from PIL import Image
import numpy as np


class YoloBestModel:
    def __init__(self, model_path):
        self.model = _YOLO(model_path)

    def predict(self, image):
        if isinstance(image, str):
            results = self.model(image)
        else:
            if isinstance(image, Image.Image):
                img = np.array(image.convert("RGB"))
            else:
                img = image  # assume already a numpy RGB array
            results = self.model(img)

        return results

    def predict_and_plot(self, image):
        """
        Runs inference and returns the annotated image (H×W×3 numpy array).
        """
        results = self.predict(image)
        annotated = results[0].plot()
        return annotated

    def get_predictions(self, image):
        """
        Returns a list of dicts with box coords, class IDs, confidences, and labels.
        """
        results = self.predict(image)
        preds = []
        for box in results[0].boxes:
            preds.append(
                {
                    "xyxy": box.xyxy.tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "label": self.model.names[int(box.cls[0])],
                }
            )
        return preds
