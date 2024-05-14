from unittest.mock import MagicMock
from unittest.mock import patch

from app.iris.services import IrisPredictionService


@patch("keras.models.load_model")
def test_iris_prediction_service_load_model(load_model):
    IrisPredictionService()
    load_model.assert_called_once_with("/app/data/iris/iris_model.keras")


@patch("keras.models.load_model")
def test_iris_prediction(load_model):
    model = MagicMock()
    result = MagicMock()
    model.return_value = result
    load_model.return_value = model
    service = IrisPredictionService()
    service.predict("data")
    model.assert_called_once_with("data")
    result.numpy.assert_called_once()


@patch("keras.models.load_model")
def test_predict_batch(load_model):
    model = MagicMock()
    result = MagicMock()
    model.return_value = result
    load_model.return_value = model
    service = IrisPredictionService()
    service.predict_batch("data")
    model.assert_called()
    result.numpy.assert_called()
