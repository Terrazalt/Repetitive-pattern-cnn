from roboflow import Roboflow
import os
from dotenv import load_dotenv

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace(os.getenv("WORKSPACE_ID")).project(os.getenv("PROJECT_ID"))
version = project.version(2)
dataset = version.download("yolov11")
