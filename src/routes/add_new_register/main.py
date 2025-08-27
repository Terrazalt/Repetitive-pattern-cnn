from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Form
from pydantic import BaseModel
from src.utils.yolo.main import YoloBestModel
from typing import Annotated, List
import os
from dotenv import load_dotenv
from PIL import Image
from roboflow import Roboflow
import json

load_dotenv()

router = APIRouter()


class Image(BaseModel):
    id: int
    file_name: str
    width: int
    height: int


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    iscrowd: int
    label: str


class Category(BaseModel):
    id: int
    name: str


class CocoBoundingBoxes(BaseModel):
    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]


class BodySchema(BaseModel):
    image: UploadFile = File(...)
    COCO_json: CocoBoundingBoxes


@router.post("")
async def detect(
    image: UploadFile = File(...),
    COCO_json: str = Form(...),
    authorization: str = Header(None, alias="Authorization"),
):
    # --- API key guard ---
    print(os.getenv("CNN_API_KEY"))
    print(os.getenv("ROBOFLOW_API_KEY"))
    print(os.getenv("PROJECT_ID"))
    print(os.getenv("WORKSPACE_ID"))
    api_key = os.getenv("CNN_API_KEY")
    if api_key:
        if not authorization:
            raise HTTPException(401, "Authorization header missing")
        if authorization != f"Bearer {api_key}":
            raise HTTPException(403, "Invalid API key")
    else:
        raise HTTPException(403, "API key is required")

    # --- Parsear COCO_json a tu modelo Pydantic ---
    try:
        CocoBoundingBoxes(**json.loads(COCO_json))
    except Exception as e:
        raise HTTPException(400, f"Invalid COCO JSON: {str(e)}")

    # Aquí puedes guardar la imagen y el JSON a disco si quieres
    # o puedes pasar el contenido directo al SDK de Roboflow

    # Ejemplo: guardar el archivo temporalmente
    image_path = f"/tmp/{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Guardar el JSON COCO individual para esta imagen (si Roboflow lo requiere por imagen)
    coco_json_path = image_path.replace(".jpg", ".json").replace(".png", ".json")
    with open(coco_json_path, "w") as f:
        json.dump(json.loads(COCO_json), f)

    try:
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        workspace = (
            rf.workspace()
        )  # Ojo, a veces necesita el nombre: rf.workspace("tu_workspace")
        project = workspace.project(os.getenv("PROJECT_ID"))
        upload_result = project.upload(
            image_path=image_path,
            annotation_path=coco_json_path,
            split="train",  # o "valid" o "test"
        )
    except Exception as e:
        # Imprime el error real (si es HTML, el status code ayuda)
        import traceback

        print("Roboflow Upload Error:", e)
        traceback.print_exc()
        raise HTTPException(500, f"Roboflow upload failed: {str(e)}")

    return {"detail": "Image uploaded to Roboflow", "roboflow_response": upload_result}
