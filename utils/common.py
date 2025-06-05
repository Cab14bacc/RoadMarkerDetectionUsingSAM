import cv2
import os
import yaml
import numpy as np
from mapJson.mapObjData import MapJsonObj

def num_to_pixel_cm(num, pixel_cm):
    return num / pixel_cm

def analyze_line_mask(mask_image, ratio=10, pixel_cm=5, index=None):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:

        mask = np.zeros_like(mask_image)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        area = cv2.contourArea(cnt)

        # measure thickness (distance to nearest background pixel)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # distanceTransform give half width
        max_thickness = dist.max() * 2

        if not check_mask_is_line(mask, pixel_cm, 250):
            return False

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        rect = cv2.minAreaRect(cnt)
        (center), (width, height), angle = rect

        # aspect ratio
        min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0
        
        # Check if the object is long and thin
        if min_aspect_ratio < ratio:
            return False
        count += 1
    return True

def get_line_width(mask):
    # measure thickness (distance to nearest background pixel)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # distanceTransform give half width
    max_thickness = dist.max() * 2
    return max_thickness

def check_mask_is_line(mask, pixel_cm, thickness=50):
    max_thickness = get_line_width(mask)
    # according to most of the road line width, smaller than 50
    if max_thickness > thickness/pixel_cm:
        return False
    return True

def analyze_all_line_mask(mask_image, ratio=10, pixel_cm=5, index=None, max_area_threshold=41000, min_area_threshold=1000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
    keep_mask_flag = False
    # create same size zero like mask
    mask = np.zeros_like(mask_image, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if (area > max_area_threshold or area < min_area_threshold):
            continue

        # draw the mask
        temp_mask = np.zeros_like(mask_image, dtype=np.uint8)
        temp_mask[labels == i] = 255

        # check line thickness
        if not check_mask_is_line(temp_mask, pixel_cm, 16):
            continue

        # Bounding rectangle
        x, y, w, h = stats[i, :4]
        aspect_ratio = float(w) / h if h != 0 else 0
        rect = cv2.minAreaRect(np.argwhere(labels == i))
        (center), (width, height), angle = rect

        # aspect ratio
        min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

        # if min_aspect_ratio < ratio:
        #     continue

        keep_mask_flag = True
        mask[labels == i] = 1
    return keep_mask_flag, mask


def clean_small_area_from_mask(mask, threshold=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep_labels = np.zeros(num_labels, dtype=np.uint8)  # label 0 is background

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= threshold:
            keep_labels[i] = 1
            
    filtered_mask = keep_labels[labels]

    filtered_mask = (filtered_mask * 255).astype(np.uint8)
    return filtered_mask

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

def binary_mask_to_array(mask):
    """
    Convert a binary mask (numpy array) to a list of coordinates where the mask is True.
    The output is a list of tuples (x, y) representing the coordinates of the True values in the mask.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError("Input mask must be a numpy array.")
    
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D binary mask.")

    ys, xs = np.where(mask)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]

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

def usage_mapping(usage, config=None):
    color_dict = { 'default': 'default', 'yellow': 'yellow', 'arrow': 'white', 'line': 'default', 'square': 'default' }
    if config is not None:
        usage_list = config.get(field='Common')['usage_list']
        if usage not in usage_list:
            print(f"Usage {usage} not in {usage_list}, use default")
            usage = 'default'
    else:
        usage_list = ['default', 'square', 'yellow', 'arrow', 'line']
        if usage not in usage_list:
            print(f"Usage {usage} not in {usage_list}, use default")
            usage = 'default'
    
    return usage