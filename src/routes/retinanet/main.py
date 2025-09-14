from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
import os
from dotenv import load_dotenv

from src.utils.retinanet.main import RetinaNetModel


load_dotenv()

router = APIRouter()
retinanet = RetinaNetModel(
    "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "best_models/retinanet/model_final.pth"
)


@router.post("/detect")
async def detect(
    image: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization"),
):
    # --- optional API-key guard ---
    api_key = os.getenv("CNN_API_KEY")
    if api_key:
        if not authorization:
            raise HTTPException(401, "Authorization header missing")
        if authorization != f"Bearer {api_key}":
            raise HTTPException(403, "Invalid API key")
    # --------------------------------

    # 1. Read bytes → PIL
    contents = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    # 2. Inference + annotation
    annotated_np: np.ndarray = retinanet.predict_and_plot(pil_img)

    # 3. Numpy → PIL → in-memory PNG
    annotated_pil = Image.fromarray(annotated_np.astype("uint8"))
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    buf.seek(0)

    # 4. Stream back to client
    return StreamingResponse(buf, media_type="image/png")


@router.post("/detect-boxes")
async def detect_boxes(
    image: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization"),
):
    api_key = os.getenv("CNN_API_KEY")
    if api_key:
        if not authorization:
            raise HTTPException(401, "Authorization header missing")
        if authorization != f"Bearer {api_key}":
            raise HTTPException(403, "Invalid API key")
    # --------------------------------

    contents = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    detections = retinanet.get_predictions(pil_img)
    return detections