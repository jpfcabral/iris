from unittest.mock import MagicMock
from unittest.mock import patch

from app.iris.tasks import predict
from app.iris.tasks import predict_batch


@patch("app.iris.tasks.IrisPredictionService")
def test_predict(IrisPredictionService):
    data = [1, 2, 3, 4]
    predict(data)
    IrisPredictionService.assert_called_once_with("data/iris/iris_model.keras")
    IrisPredictionService.return_value.predict.assert_called_once_with(data)


@patch("app.iris.tasks.IrisPredictionService")
def test_predict_batch(IrisPredictionService):
    model_id = "model_id"
    data = [[1, 2, 3, 4]]
    predict_batch(model_id, data)
    IrisPredictionService.assert_called_once_with(model_id)
    IrisPredictionService.return_value.predict_batch.assert_called_once_with(data)
