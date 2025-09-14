from fastapi import APIRouter
from roboflow import Roboflow
import os
from dotenv import load_dotenv
import shutil
from datetime import datetime
import json

# detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

router = APIRouter()
load_dotenv()


@router.get("/retrain")
def retrain_model():
    """
    Retrains the RetinaNet model with the latest data from Roboflow.
    """
    # --- 1. Backup existing model ---
    model_path = "best_models/retinanet/model_final.pth"
    if os.path.exists(model_path):
        shutil.copy(
            model_path,
            f"best_models/retinanet/model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        )

    # --- 2. Download dataset from Roboflow ---
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("WORKSPACE_ID")).project(os.getenv("PROJECT_ID"))
    dataset = project.version(4).download("coco")  # As per user instruction
    data_path = dataset.location

    # --- 3. Register datasets for detectron2 ---
    # Unregister if they already exist to prevent errors on re-runs
    for d in ["mi_dataset_train", "mi_dataset_val"]:
        if d in DatasetCatalog.list():
            DatasetCatalog.remove(d)
        if d in MetadataCatalog.list():
            MetadataCatalog.remove(d)

    register_coco_instances(
        "mi_dataset_train", {}, f"{data_path}/train/_annotations.coco.json", f"{data_path}/train"
    )
    register_coco_instances(
        "mi_dataset_val", {}, f"{data_path}/valid/_annotations.coco.json", f"{data_path}/valid"
    )

    # --- 4. Get number of classes and calculate iterations ---
    with open(f"{data_path}/train/_annotations.coco.json") as f:
        coco_data = json.load(f)
    num_classes = len(coco_data["categories"])

    train_images_path = os.path.join(data_path, "train")
    n_images = len([f for f in os.listdir(train_images_path) if f.endswith(".jpg")])

    # --- 5. Configure training ---
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    )

    # Use existing model as checkpoint, otherwise use COCO pre-trained weights
    if os.path.exists(model_path):
        cfg.MODEL.WEIGHTS = model_path
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
        )

    cfg.DATASETS.TRAIN = ("mi_dataset_train",)
    cfg.DATASETS.TEST = ("mi_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025

    epochs = 10  # As per user's example script
    max_iter = (n_images // cfg.SOLVER.IMS_PER_BATCH) * epochs
    cfg.SOLVER.MAX_ITER = max_iter

    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = "./output_retinanet"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- 6. Train the model ---
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # --- 7. Save the newly trained model ---
    shutil.copy(
        os.path.join(cfg.OUTPUT_DIR, "model_final.pth"),
        model_path,
    )

    # --- 8. Clean up ---
    shutil.rmtree(data_path)  # remove downloaded dataset
    shutil.rmtree(cfg.OUTPUT_DIR)  # remove output directory

    return {"message": "RetinaNet model retraining completed successfully."}