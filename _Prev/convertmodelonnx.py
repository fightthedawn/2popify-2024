import tensorflow as tf
import tf2onnx
import onnx

# Load your existing TensorFlow model
tf_model = tf.saved_model.load('/Users/giovonnilobato/Documents/GitHub/2popify-2024/ModelsTrained/2popmodel100000-20240110')

# Convert the model to ONNX
input_signature = [tf.TensorSpec([None, 24000], tf.float32, name='input')]  # Adjust the shape based on your model's input
onnx_model, _ = tf2onnx.convert.from_function(
    tf_model.__call__,
    input_signature,
    opset=13,
    output_path="2pop_model.onnx"
)

print("Model converted and saved as 2pop_model.onnx")