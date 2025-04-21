# RoadMarkerDetectionUsingSAM
Test for UAV or top down view street image data. Detect basic road marker on road.
Implement base on meta segment anything.

# Prepare
Download checkpoints `sam_vit_h_4b8939.pth` from meta segment anything
change checkpoints path in config.yml

# config parameters
- ColorFilter (Keep the color we need as road marker)
    - color_list: The color that needs to be retained as the mask image
    - color_threshold: Threshold determines how close the colors are
    - area_threshold: The noise area size threshold to be removed

- Predictor (meta sam predict)
    - checkpoint: checkpoint path
    - area_threshold: Maximum mask size to be retained

# Usage
Predict each element on road
`python predictor.py -i <input_image> -o <output_path>`

Test close color filter on road
`python color_filter.py -i <input_image> -o <output_path>`