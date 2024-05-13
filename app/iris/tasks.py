from typing import List

from celery import Celery

from app.config import settings
from app.iris.models import IrisEnum
from app.iris.services import IrisPredictionService

app = Celery(
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_BACKEND_URL,
    accept_content=["pickle", "json"],
)


@app.task(name="iris.predict")
def predict(data: List[float]) -> List[float]:
    """
    Predict a batch of iris flowers

    Args:
        data (List[float]): A list of iris flower parameters

    Returns:
        List[float]: A list of predictions
    """
    prediction_service = IrisPredictionService("data/iris/iris_model.keras")
    return prediction_service.predict(data)


@app.task(name="iris.predict_batch", serializer="pickle")
def predict_batch(data: List[List[float]]) -> List[List[float]]:
    """
    Predict a batch of iris flowers

    Args:
        data (List[List[float]]): A list of iris flowers

    Returns:
        List[List[float]]: A list of predictions
    """
    results = []
    prediction_service = IrisPredictionService("data/iris/iris_model.keras")

    predictions = prediction_service.predict_batch(data)

    species = [IrisEnum.from_prediction(prediction) for prediction in predictions]

    for specie, prediction in zip(species, predictions):
        results.append(
            {
                "specie": specie.name,
                "logits": prediction.tolist(),
            }
        )

    return results
