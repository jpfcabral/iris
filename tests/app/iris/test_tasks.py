from unittest.mock import MagicMock
from unittest.mock import patch

from app.iris.tasks import predict
from app.iris.tasks import predict_batch
from app.iris.tasks import train_iris_model


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


@patch("app.iris.tasks.mlflow")
@patch("app.iris.tasks.pd")
@patch("app.iris.tasks.train_test_split")
@patch("app.iris.tasks.keras")
def test_train_iris_model(keras, train_test_split, pd, mlflow):
    train_test_split.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    model = MagicMock()
    model.evaluate.return_value = (0.2, 0.3)
    keras.models.Sequential.return_value = model
    train_iris_model("data")
