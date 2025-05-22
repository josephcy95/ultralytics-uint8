import os
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path

def convert_to_uint8_tflite(model_path, output_dir="uint8_models"):
    """
    Convert a YOLOv8 model to uint8 TFLite format.
    
    Args:
        model_path (str): Path to the YOLOv8 model (.pt file)
        output_dir (str): Directory to save the converted model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and export to TFLite with int8 quantization first
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get model name without extension
    model_name = Path(model_path).stem
    
    print("Exporting to TFLite with int8 quantization...")
    model.export(format="tflite", int8=True)
    
    # Path to the saved model directory
    saved_model_dir = f"{model_name}_saved_model"
    
    print("Converting to uint8 TFLite format...")
    # Load the quantized TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Configure for uint8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.uint8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the uint8 model
    output_path = os.path.join(output_dir, f"{model_name}_uint8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Model successfully converted and saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLOv8 model to uint8 TFLite format")
    parser.add_argument("model_path", type=str, help="Path to the YOLOv8 model (.pt file)")
    parser.add_argument("--output-dir", type=str, default="uint8_models", help="Directory to save the converted model")
    
    args = parser.parse_args()
    convert_to_uint8_tflite(args.model_path, args.output_dir) 