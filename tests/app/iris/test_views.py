from unittest.mock import patch

from app.iris.models import IrisEnum
from app.iris.models import IrisPredictionRequest
from app.iris.views import get_batch_prediction
from app.iris.views import predict_batch_iris_data
from app.iris.views import predict_single_iris_data
from app.iris.views import training_model


@patch("app.iris.views.IrisEnum")
@patch("app.iris.views.IrisPredictionService")
def test_predict(IrisPredictionService, IrisEnum):
    data = IrisPredictionRequest(sepal_length=1, sepal_width=2, petal_length=3, petal_width=4)
    response = predict_single_iris_data(data)
    IrisPredictionService.assert_called_once_with("/app/data/iris/iris_model.keras")


@patch("app.iris.views.predict_batch")
def test_predict_batch(
    predict_batch,
):
    data = [IrisPredictionRequest(sepal_length=1, sepal_width=2, petal_length=3, petal_width=4)]
    predict_batch_iris_data(data)
    predict_batch.apply_async.assert_called()


@patch("app.iris.views.predict_batch")
def test_get_batch_prediction(predict_batch):
    task_id = "task_id"
    get_batch_prediction(task_id)
    predict_batch.AsyncResult.assert_called_once_with(task_id)


@patch("app.iris.views.train_iris_model")
def test_training_model(train_iris_model):
    training_model("data")
    train_iris_model.apply_async.assert_called()
