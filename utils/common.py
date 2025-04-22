import cv2
import os
import yaml
import numpy as np

def clean_small_area_from_mask(mask, threshold=50):
    # contour and connectedComponent seems to be the same result
    # keep contour method for maybe future use on shape

    # Apply connected components
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # for i in range(1, num_labels):
    #     x, y, w, h, area = stats[i]
    #     if area < 100:  # filter out small blobs
    #         mask[labels == i] = 0
    # return mask

    # Remove objects that are too small or have odd aspect ratios that don’t match typical road markers:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < threshold:  # filter out small blobs
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    return mask

# load config file return config
def load_config(config_path, field=None):
    # check if config_path exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if field is None:
        return config
    else:
        if field not in config:
            raise KeyError(f"Field '{field}' not found in config file: {config_path}")
        return config[field]

# check mask shape is close to square and clean small size
def square_shape_compare(mask, min_size_threshold=10000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create a black image same size as image
    height, width = mask.shape

    black_image = np.zeros((height, width), dtype=np.uint8)

    for i, contour in enumerate(contours):
        # skip small area
        area_size = cv2.contourArea(contour)
        if area_size < min_size_threshold:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        # draw the contour and approx on the black image
        if (approx.shape[0] == 4):
            cv2.drawContours(black_image, [approx], -1, (255, 255, 255), thickness=cv2.FILLED)
        
        return black_image
    return mask