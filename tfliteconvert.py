import tensorflow as tf
import os
import sys

def convert_model(input_folder, output_folder):
    # Try to load the model using the TensorFlow SavedModel API
    try:
        model = tf.saved_model.load(input_folder)
    except Exception as e:
        print("Error loading model using TensorFlow SavedModel API:", e)
        sys.exit(1)

    # Try to convert the model to TensorFlow Lite
    try:
        # If the model contains a signature, specify which one to use
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()

        # Create the export folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the TensorFlow Lite model to the export folder
        tflite_model_path = os.path.join(output_folder, "converted_model.tflite")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model converted and saved to {tflite_model_path}")
    except Exception as e:
        print("Error converting model to TensorFlow Lite:", e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_tflite.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_model(input_folder, output_folder)