from typing import List

import keras


class IrisPredictionService:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)

    def predict(self, data: List[List[float]]):
        result = self.model(data)
        return result.numpy()

    def predict_batch(self, data: List[List[float]]) -> List[List[float]]:
        result_list = []

        for flower in data:
            result = self.predict([flower])
            result_list.append(result)

        return result_list
