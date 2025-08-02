from abc import ABC, abstractmethod
from app.slices.prediction.models.scene import Scene
from app.slices.prediction.models.prediction import Prediction


class PredictionService(ABC):
    @abstractmethod
    def predict(self, scene: Scene) -> Prediction:
        pass