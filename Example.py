# using ChebyKAKeras implement a simple model for MNIST
import tensorflow as tf
from tensorflow.keras.layers import Input, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.datasets import mnist
from ChebyKANKeras import ChebyKANLayer


if __name__ == "__main__":
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    # Preprocess the data
    model = Sequential(
        [
            Input(
                shape=(28 * 28,)
            ),  # Input layer with the shape of flattened MNIST images
            ChebyKANLayer(28 * 28, 32, 4),  # First ChebyKAN layer
            LayerNormalization(),  # Layer normalization
            ChebyKANLayer(32, 16, 4),  # Second ChebyKAN layer
            LayerNormalization(),  # Layer
            ChebyKANLayer(16, 10, 4),  # Third ChebyKAN layer
        ]
    )

    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )

    # Print the model summary
    model.summary()

    history = model.fit(
        x_train, y_train, epochs=10, batch_size=32, validation_split=0.2
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Predict the labels for the test set
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    confusion_matrix = tf.math.confusion_matrix(y_test, y_pred_classes)

    # Calculate precision, recall, and F1 score
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    precision.update_state(y_test, y_pred_classes)
    recall.update_state(y_test, y_pred_classes)

    precision_value = precision.result().numpy()
    recall_value = recall.result().numpy()
    f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Precision: {precision_value*100:.2f}%")
    print(f"Recall: {recall_value*100:.2f}%")
    print(f"F1 Score: {f1_score*100:.2f}%")
