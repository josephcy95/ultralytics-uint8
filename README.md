# YOLOv8 to uint8 TFLite Converter

This script converts YOLOv8 models to uint8 quantized TFLite format, which is compatible with Frigate's input format.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can use the script in two ways:

### 1. Command Line Interface

```bash
python convert_to_uint8_tflite.py path/to/your/model.pt --output-dir uint8_models
```

### 2. Python API

```python
from convert_to_uint8_tflite import convert_to_uint8_tflite

# Convert your model
model_path = "path/to/your/model.pt"
output_path = convert_to_uint8_tflite(model_path, output_dir="uint8_models")
```

## Output

The script will:
1. Convert your YOLOv8 model to TFLite format with int8 quantization
2. Convert the int8 model to uint8 format
3. Save the final uint8 model in the specified output directory

The output model will be named `{model_name}_uint8.tflite` and will be compatible with Frigate's uint8 input format.

## Notes

- The conversion process requires TensorFlow and the TFLite runtime
- The script creates a temporary saved model directory during conversion
- The final model will be saved in the specified output directory
