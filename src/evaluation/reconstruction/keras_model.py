import os

if os.environ.get("BLINKWISE_USE_TFMOT_KERAS", "0") == "0":
    from tensorflow import keras
else:
    from tensorflow_model_optimization.python.core.keras.compat import keras

from .base import CurveReconstructor

class KerasCurveReconstructor(CurveReconstructor):
    """
    Curve reconstructor using a Keras model.
    """
    def _load_model(self):
        return keras.models.load_model(self.checkpoint_path, compile=False)
