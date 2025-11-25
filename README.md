# RoadMarkerDetectionUsingSAM

Test for UAV or top down view street image data. Detect basic road marker on road.
Implement base on meta segment anything.

## Installation

This project depends on PyTorch. Please install it first by following the official instructions for your specific hardware (CPU or GPU with CUDA) on the [PyTorch website](https://pytorch.org/get-started/locally/).

For example, for a recent version of CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Or for CPU only:

```bash
pip install torch torchvision torchaudio
```

Once PyTorch is installed, you can install this project in editable mode:

```bash
pip install -e .
```

## Prepare

Download checkpoints `sam_vit_h_4b8939.pth` from meta segment anything
change checkpoints path in config.yml

## Config Parameters

- ColorFilter (Keep the color we need as road marker)
  - color_list: The color that needs to be retained as the mask image
  - color_threshold: Threshold determines how close the colors are
  - area_threshold: The noise area size threshold to be removed

- Predictor (meta sam predict)
  - checkpoint: checkpoint path
  - area_threshold: Maximum mask size to be retained

## Usage
`import RoadMarkerDetectionUsingSAM`
see pipeline.py for example usage

<br>
Or directly use each component:

Predict each element on road
`python -m predictor.py -i <input_image> -o <output_path>`

Test close color filter on road
`python -m color_filter.py -i <input_image> -o <output_path>`
