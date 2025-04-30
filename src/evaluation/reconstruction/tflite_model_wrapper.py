import numpy as np

class TFLiteModelWrapper:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        # Store quantization parameters
        self.input_scale = self.input_details[0]['quantization_parameters']['scales'][0]
        self.input_zero_point = self.input_details[0]['quantization_parameters']['zero_points'][0]
        self.output_scale = self.output_details[0]['quantization_parameters']['scales'][0]
        self.output_zero_point = self.output_details[0]['quantization_parameters']['zero_points'][0]

    def quantize_input(self, x):
        """Quantize float32 input to uint8"""
        return np.clip(
            np.round(x / self.input_scale + self.input_zero_point),
            0,
            255
        ).astype(np.uint8)

    def dequantize_output(self, x):
        """Dequantize uint8 output to float32"""
        return (x.astype(np.float32) - self.output_zero_point) * self.output_scale

    def predict(self, dataset, verbose=0):
        # Handle batched dataset
        predictions = []
        for batch in dataset:
            if isinstance(batch, tuple):
                x = batch[0]  # If dataset returns (x, y) pairs
            else:
                x = batch

            # Quantize input
            x_quantized = self.quantize_input(x)

            for i in range(len(x_quantized)):
                self.interpreter.set_tensor(self.input_details[0]['index'], x_quantized[i:i + 1])
                self.interpreter.invoke()
                quantized_output = self.interpreter.get_tensor(self.output_details[0]['index'])
                batch_pred = self.dequantize_output(quantized_output)
                predictions.append(batch_pred)

        return np.vstack(predictions)
