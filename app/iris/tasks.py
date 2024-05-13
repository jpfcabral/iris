from typing import List

import keras
import mlflow.keras
import numpy as np
import pandas as pd
from celery import Celery
from loguru import logger
from sklearn.model_selection import train_test_split

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


@app.task(name="iris.training", serializer="pickle")
def train_iris_model(data_path: str):
    """
    Train the iris model

    Args:
        data_path (str): The path to the iris csv data
    """

    logger.info(f"Training model with data from {data_path}")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.keras.autolog()

    column_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
    iris_data = pd.read_csv(data_path, names=column_names)

    Y = iris_data["Species"]
    X = iris_data.drop(["Species"], axis=1)

    label_results = []
    Y_encoded = []

    for y in Y:
        if y == "Iris-setosa":
            label_results.append([1, 0, 0])
            Y_encoded.append(0)
        elif y == "Iris-versicolor":
            label_results.append([0, 1, 0])
            Y_encoded.append(1)
        else:
            label_results.append([0, 0, 1])
            Y_encoded.append(2)

    Y_final = np.array(label_results)
    Y_encoded = np.array(Y_encoded)

    seed = 42
    np.random.seed(seed)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y_final, test_size=0.25, random_state=seed, stratify=Y_encoded, shuffle=True
    )

    x_train_new = x_train
    x_test_new = x_test

    model = keras.models.Sequential()
    model.add(keras.Input(shape=(4,)))
    model.add(
        keras.layers.Dense(
            10,
            activation=keras.activations.relu,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(
        keras.layers.Dense(
            7,
            activation=keras.activations.relu,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(
        keras.layers.Dense(
            5,
            activation=keras.activations.relu,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
        )
    )
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train_new, y_train, epochs=700, batch_size=7)

    loss, accuracy = model.evaluate(x_test_new, y_test)

    mlflow.keras.log_model(model, "iris_model")

    return {"loss": loss, "accuracy": accuracy, "model_path": "/app/data/iris/iris_model2.keras"}
