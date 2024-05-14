from typing import List

import keras
import mlflow.keras

from app.config import settings

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


class IrisPredictionService:
    def __init__(self, model_path: str = None):
        """
        Initialize the Iris prediction service

        Args:
            model_path (str): The path to the model
        """

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        """
        Load the model

        Args:
            model_path (str): The path to the model
        """
        if not model_path:
            return keras.models.load_model("/app/data/iris/iris_model.keras")

        return mlflow.keras.load_model(model_path)

    def predict(self, data: List[List[float]]):
        """
        Predict the Iris species

        Args:
            data (List[List[float]]): A list of Iris flowers
        """
        result = self.model(data)
        return result.numpy()

    def predict_batch(self, data: List[List[float]]) -> List[List[float]]:
        """
        Predict a batch of Iris flowers

        Args:
            data (List[List[float]]): A list of Iris flowers
        """
        result_list = []

        for flower in data:
            result = self.predict([flower])
            result_list.append(result)

        return result_list
