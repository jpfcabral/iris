from typing import List

import numpy as np
from celery.result import AsyncResult
from fastapi import APIRouter
from loguru import logger

from app.iris.models import IrisBatchPredictionRequest
from app.iris.models import IrisEnum
from app.iris.models import IrisPredictionRequest
from app.iris.services import IrisPredictionService
from app.iris.tasks import predict_batch
from app.iris.tasks import train_iris_model

router = APIRouter()


@router.post("/predict")
def predict_single_iris_data(request: IrisPredictionRequest):
    """Predict a single iris flower"""
    input = np.array(
        [[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]]
    )
    prediction_service = IrisPredictionService()

    prediction = prediction_service.predict(input)

    specie = IrisEnum.from_prediction(prediction)

    return {
        "specie": specie.name,
        "logits": prediction.tolist(),
    }


@router.post("/predict/batch")
def predict_batch_iris_data(request: IrisBatchPredictionRequest):
    """Predict a batch of iris flowers"""
    data = request.data
    input = [
        np.array(
            [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]]
        )
        for flower in data
    ]

    logger.info(f"Predicting {len(input)} flowers: {input}")

    async_result: AsyncResult = predict_batch.apply_async(
        args=(
            request.model_id,
            input,
        )
    )
    task_id = async_result.id

    return {"task_id": task_id}


@router.get("/predict/batch/{task_id}")
def get_batch_prediction(task_id: str):
    """Get the result of a batch prediction"""
    async_result: AsyncResult = predict_batch.AsyncResult(task_id)

    if async_result.ready():
        return async_result.get()

    return {"status": "PENDING"}


@router.post("/training")
def training_model(data_path: str = "/app/data/iris/iris.data"):
    """Train the model with the given params"""
    result = train_iris_model.apply_async(args=(data_path,))
    task_id = result.id

    return {"task_id": task_id}
