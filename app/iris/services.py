from typing import List

import keras
import mlflow.keras

from app.config import settings

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


class IrisPredictionService:
    def __init__(self, model_path: str = None):

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        if not model_path:
            return keras.models.load_model("/app/data/iris/iris_model.keras")

        return mlflow.keras.load_model(model_path)

    def predict(self, data: List[List[float]]):
        result = self.model(data)
        return result.numpy()

    def predict_batch(self, data: List[List[float]]) -> List[List[float]]:
        result_list = []

        for flower in data:
            result = self.predict([flower])
            result_list.append(result)

        return result_list
