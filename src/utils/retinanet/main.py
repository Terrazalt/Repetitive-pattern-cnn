from PIL import Image
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


class RetinaNetModel:
    def __init__(self, config_file: str, model_path: str, confidence_threshold: float = 0.5):
        """
        Initializes the RetinaNet model.
        """
        setup_logger()
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def predict(self, image):
        """
        Runs inference on a single image.
        `image` can be a PIL Image or a numpy array (H, W, 3) in RGB format.
        """
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image  # assume already a numpy RGB array

        # detectron2 expects BGR
        bgr_img = img[:, :, ::-1]
        outputs = self.predictor(bgr_img)
        return outputs

    def predict_and_plot(self, image):
        """
        Runs inference and returns the annotated image (H×W×3 numpy array).
        """
        outputs = self.predict(image)

        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image

        v = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        annotated_image_bgr = out.get_image()
        # Convert BGR to RGB for response
        return annotated_image_bgr[:, :, ::-1]

    def get_predictions(self, image):
        """
        Returns a list of dicts with box coords, class IDs, confidences, and labels.
        """
        outputs = self.predict(image)
        instances = outputs["instances"].to("cpu")

        preds = []
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.tolist()[0]
            preds.append(
                {
                    "xyxy": box,
                    "confidence": float(instances.scores[i]),
                    "class_id": int(instances.pred_classes[i]),
                    "label": self.metadata.thing_classes[int(instances.pred_classes[i])],
                }
            )
        return preds
