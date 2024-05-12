import keras
import numpy as np
from fastapi import APIRouter

from app.iris.models import IrisEnum
from app.iris.models import IrisPredictionRequest
from app.iris.services import IrisPredictionService

router = APIRouter()


@router.post("/predict")
def predict_single_iris_data(request: IrisPredictionRequest):
    input = np.array(
        [[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]]
    )
    prediction_service = IrisPredictionService("/app/data/iris/iris_model.keras")

    prediction = prediction_service.predict(input)

    specie = IrisEnum.from_prediction(prediction)

    return {
        "specie": specie.name,
        "logits": prediction.tolist(),
    }
