import cv2
import os
import yaml
import numpy as np

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
    # temp = np.zeros_like(dist)
    # temp = cv2.normalize(dist, temp, 0, 255, cv2.NORM_MINMAX)
    # temp = temp.astype(np.uint8)
    # cv2.imshow("sdlfkjas", temp)
    # cv2.waitKey()
    # distanceTransform give half width
    max_thickness = dist.max() * 2
    return max_thickness

def check_mask_is_line(mask, pixel_cm, thickness_threshold=50):
    
    mask = np.pad(mask, ((10, 10), (10, 10))) # pad with zero to handle truncated components at borders
    max_thickness = get_line_width(mask)
    # according to most of the road line width, smaller than 50

    if max_thickness > thickness_threshold/pixel_cm:
        return False
    return True

def analyze_all_line_mask(mask_image, ratio=10, pixel_cm=5, index=None, max_area_threshold=41000, min_area_threshold=1000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_image, connectivity=8)

    # max_label = np.max(labels)
    # color_interval = 255 // max_label
    # temp = np.array(labels, dtype=np.uint8) * color_interval
    # hsv_img = np.zeros((*temp.shape, 3), dtype=np.uint8)
    # hsv_img[..., 0] = temp
    # hsv_img[..., 1] = 255
    # hsv_img[..., 2] = 255
    # bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


    keep_mask_flag = False
    # create same size zero like mask
    mask = np.zeros_like(mask_image, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # left = stats[i, cv2.CC_STAT_LEFT]
        # top = stats[i, cv2.CC_STAT_TOP]    
        # cv2.putText(bgr_img, str(area), (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if (area > max_area_threshold or area < min_area_threshold):
            continue


        # draw the mask
        temp_mask = np.zeros_like(mask_image, dtype=np.uint8)
        temp_mask[labels == i] = 255

        # check line thickness
        # if not check_mask_is_line(temp_mask, pixel_cm, 16):
        if not check_mask_is_line(temp_mask, pixel_cm, 18):
            continue

        # Bounding rectangle
        # x, y, w, h = stats[i, :4]
        # aspect_ratio = float(w) / h if h != 0 else 0
        # rect = cv2.minAreaRect(np.argwhere(labels == i))
        # (center), (width, height), angle = rect

        # # aspect ratio
        # min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

        # if min_aspect_ratio < ratio:
        #     continue

        keep_mask_flag = True
        mask[labels == i] = 1

        max_label = np.max(labels)

    # cv2.imshow("HSV as BGR", bgr_img)
    # cv2.imshow("sdlkfddj", mask * 255)
    # cv2.waitKey()

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

def enhance_edge(image):
    """
    Enhance edges in the input image using Canny edge detection and overlay the edges.
    Returns an image with enhanced edges.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, 100, 200)

    # If input is color, overlay edges in red
    if len(image.shape) == 3:
        edge_img = image.copy()
        edge_img[edges > 0] = [0, 0, 255]  # Red edges
    else:
        # For grayscale, just return the edges
        edge_img = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

    return edge_img

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

def connected_components_from_coordinates(coordinate_set, connectivity=8):
    from collections import deque

    visited_points = set()
    connected_components = []

    neighbors_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors_list += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for point in coordinate_set:
        if point in visited_points:
            continue
        
        component = []
        queue = deque([point])
        visited_points.add(point)

        while queue:
            current_point = queue.popleft()
            component.append(current_point)

            x, y = current_point
            for dx, dy in neighbors_list:
                neighbors_point = (x + dx, y + dy)
                if neighbors_point in coordinate_set and neighbors_point not in visited_points:
                    visited_points.add(neighbors_point)
                    queue.append(neighbors_point)

        connected_components.append(component)
    return connected_components

def connected_components_to_scaled_mask(connected_components, original_shape, scaled):
    height, width = original_shape
    height = height * scaled
    width = width * scaled
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)

    original_height, original_width = original_shape
    for component in connected_components:
        for point in component:
            x, y = point
            result_height = min(int(y/original_height * height), height - 1)
            result_width = min(int(x/original_width * width), width - 1)
            mask[int(result_height), int(result_width)] = 1
    
    return mask.astype(np.uint8) * 255

def get_image(input_path):
    if isinstance(input_path, str):
        # check input_path is file
        if not os.path.isfile(input_path):
            raise ValueError(f"Input path {input_path} is not a valid file.")
        image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # image = cv2.imread(input_path)

    elif isinstance(input_path, np.ndarray):
        image = input_path
    else:
        raise TypeError("mask_image should be either a file path or a numpy array")
    return image