from fastapi import FastAPI, Path, Query, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shutil
import time
from starlette import status
import onnxruntime
import detector
from typing import List
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN


onnx_model_path = 'models/faster_rcnn.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load('models/faster_rcnn_sm_epoch_5.pth', map_location=device))

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="files"), name="static")


@app.post("/upload/")
async def create_upload_file(uploaded_file: UploadFile):
    file_ext = uploaded_file.filename.split(".")[-1]
    filename = f"{round(time.time() * 1000)}.{file_ext}"
    path = f"files/{filename}"
    with open(path, 'w+b') as file:
        shutil.copyfileobj(uploaded_file.file, file)

    return {
        'file': filename,
        'content': uploaded_file.content_type,
        'path': f"static/{filename}",
    }


class PredictionRequest(BaseModel):
    filename: str = Field(min_length=5)

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "1293912839123.jpg",
            }
        }


class Person:
    boxes: List[List[float]]
    scores: List[float]
    latency: float
    output: str

    def __init__(self, boxes: List[List[float]], scores: List[float], latency: float, output: str):
        self.boxes = boxes
        self.scores = scores
        self.latency = latency
        self.output = output


@app.post("/predict/pytorch", status_code=status.HTTP_200_OK)
async def predict(prediction_request: PredictionRequest):
    filename = prediction_request.filename
    boxes, scores, latency, output = detector.predict(image_name=filename, model=model, device=device)
    return Person(boxes=[box.tolist() for box in boxes], scores=[score.tolist() for score in scores], latency=latency, output=output)


@app.post("/predict/onnx", status_code=status.HTTP_200_OK)
async def predict(prediction_request: PredictionRequest):
    filename = prediction_request.filename
    boxes, scores, latency, output = detector.predict_onnx(image_name=filename, session=session)
    return Person(boxes=[box.tolist() for box in boxes], scores=[score.tolist() for score in scores], latency=latency, output=output)
