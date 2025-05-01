import cv2
import os
import yaml
import numpy as np
from mapJson.mapObjData import MapJsonObj
# input binary mask image.
# ratio is the aspect ratio threshold to determine if the object is long and thin.
def analyze_line_mask(mask_image, ratio=10, pixel_cm=5):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect = cv2.minAreaRect(cnt)
        (center), (width, height), angle = rect
            
        # most of marker and line width smaller than 250cm 
        if (min(width, height) > 250/pixel_cm):
            return False
        # aspect ratio
        min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

        # Optional: Calculate solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # Check if the object is long and thin
        if min_aspect_ratio < ratio:
            return False
    return True

def analyze_all_line_mask(mask_image, ratio=10, pixel_cm=5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
    keep_mask_flag = False
    # create same size zero like mask
    mask = np.zeros_like(mask_image, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Bounding rectangle
        x, y, w, h = stats[i, :4]
        aspect_ratio = float(w) / h if h != 0 else 0
        rect = cv2.minAreaRect(np.argwhere(labels == i))
        (center), (width, height), angle = rect
        
        # most of marker and line width smaller than 250cm 
        if (min(width, height) > 250/pixel_cm):
            continue
        # aspect ratio
        min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

        # Optional: Calculate solidity
        hull = cv2.convexHull(np.argwhere(labels == i))
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # Check if the object is long and thin
        if min_aspect_ratio < ratio:
            continue
        
        # draw the mask
        mask[labels == i] = 1
        keep_mask_flag = True
    return keep_mask_flag, mask


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

def enhance_brightness_saturation(image_BGR):

    hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 50, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white, mask_yellow)

    hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * 2.0, 0, 255), hsv[:, :, 2])
    hsv[:, :, 1] = np.where(mask > 0, np.clip(hsv[:, :, 1] * 1.3, 0, 255), hsv[:, :, 1])

    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced_img

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

def close_color(pixel, color, threshold=100):
    # calculate the distance between the pixel and the color
    distance = np.sqrt(np.sum((color - pixel) ** 2))

    # check if the distance is less than the threshold
    if distance < threshold:
        return True
    else:
        return False
    
def mask_most_of_color(image, mask, color, threshold=60, index=0):
    # get the true value index in mask
    ys, xs = np.where(mask)
    # get the color value in image
    color_value = image[ys, xs]

    # create array with shape(len(xs))
    color_mask = np.zeros(len(xs), dtype=bool)
    # check if the color_value is close to color or not using common.close_color function, as boolean array
    for i in range(len(xs)):
        color_mask[i] = close_color(color_value[i], color, threshold=threshold)
    
    # more true value in color_mask return true, else false
    if np.sum(color_mask) > (len(xs) / 10):
        return True
    else:
        return False


def build_map_json_obj(mask, index):
    # find bounding box of mask
    bbox = find_boolean_mask_bounding(mask)
    if bbox is None:
        print("No bounding box found for the mask.")
        return None

    # transform mask boolean to uint8 (0 or 1)
    mask = mask.astype(np.uint8)
    # copy bounding box area from mask to mask_bbox
    mask_bbox = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    temp_map = {}
    # check if index in temp_map, if not, set to 'other'
    label = 'other'
    if index in temp_map:
        label = temp_map[index]
    # create MapJsonObj object
    map_json_obj = MapJsonObj(label=str(label), bbox=bbox, mask=mask_bbox)
    return map_json_obj.to_dict()

def find_boolean_mask_bounding(mask):
    # Get indices where mask is True
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None  # No True values, bounding box is undefined

    # numpy to value
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    # numpy.int64 to int
    min_x, max_x = int(min_x), int(max_x)
    min_y, max_y = int(min_y), int(max_y)
    return [min_x, min_y, max_x, max_y]
