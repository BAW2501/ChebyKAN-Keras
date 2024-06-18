import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
# adapted 1 to 1 from https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
class ChebyKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize Chebyshev coefficients
        initializer = initializers.RandomNormal(mean=0.0, stddev=1 / (input_dim * (degree + 1)))
        self.cheby_coeffs = self.add_weight(
            shape=(input_dim, output_dim, degree + 1),
            initializer=initializer,
            trainable=True
        )
        # Create arange buffer
        self.arange = tf.range(0, degree + 1, 1, dtype=tf.float32)

    def call(self, inputs):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize inputs to [-1, 1] using tanh
        x = tf.tanh(inputs)
        # View and repeat input degree + 1 times
        x = tf.expand_dims(x, axis=-1)  # shape = (batch_size, input_dim, 1)
        x = tf.tile(x, [1, 1, self.degree + 1])  # shape = (batch_size, input_dim, degree + 1)
        # Apply acos
        x = tf.acos(x)
        # Multiply by arange [0 .. degree]
        x = x * self.arange
        # Apply cos
        x = tf.cos(x)
        # Compute the Chebyshev interpolation
        y = tf.einsum('bid,iod->bo', x, self.cheby_coeffs)  # shape = (batch_size, output_dim)
        return y

if __name__ == '__main__':
    # Example of how to use the layer
    input_dim = 5
    output_dim = 3
    degree = 4
    layer = ChebyKANLayer(input_dim, output_dim, degree)

    # Create a model for testing
    inputs = tf.keras.Input(shape=(input_dim,))
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Print the model summary
    model.summary()
