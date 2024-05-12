import numpy as np

from app.iris.models import IrisEnum


def test_iris_enum():
    assert IrisEnum.from_prediction(np.array([1, 0, 0])).name == "setosa"
    assert IrisEnum.from_prediction(np.array([0, 1, 0])).name == "versicolor"
    assert IrisEnum.from_prediction(np.array([0, 0, 1])).name == "virginica"
    assert str(IrisEnum.setosa) == "setosa"
