from typing import List

import keras


class IrisPredictionService:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)

    def predict(self, data: List[List[float]]):
        result = self.model(data)
        return result.numpy()
