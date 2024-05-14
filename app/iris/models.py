import enum

from pydantic import BaseModel


class IrisPredictionRequest(BaseModel):
    """Request model for Iris prediction"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisEnum(enum.Enum):
    """Enum for Iris species"""

    setosa = 0
    versicolor = 1
    virginica = 2

    @classmethod
    def from_prediction(cls, prediction):
        return cls(prediction.argmax())

    def __str__(self):
        return self.name


class IrisBatchPredictionRequest(BaseModel):
    """Request model for batch Iris prediction"""

    model_id: str = None
    data: list[IrisPredictionRequest]
