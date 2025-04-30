import tensorflow as tf
from .tflite_model_wrapper import TFLiteModelWrapper

from .base import CurveReconstructor


class TFLiteCurveReconstructor(CurveReconstructor):
    """
    Curve reconstructor using TFLite model.
    """
    def _load_model(self):
        assert "tflite" in self.checkpoint_path.suffix, "TFLite model must have .tflite extension."

        self._interpreter = tf.lite.Interpreter(model_path=self.checkpoint_path.as_posix())
        self._interpreter.allocate_tensors()
        return TFLiteModelWrapper(self._interpreter)
