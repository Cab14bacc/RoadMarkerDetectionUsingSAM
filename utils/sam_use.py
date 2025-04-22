import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything.utils.transforms import ResizeLongestSide

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def save_mask(masks, path="./"):
    # Assuming 'masks' contains the generated masks from SAM2
    for i, mask in enumerate(masks):
        # Convert the mask to a PIL Image
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to 8-bit grayscale

        # Save the mask image
        file_path = os.path.join(path, f"mask_{i}.png")
        mask_image.save(file_path)  # Save as PNG with a unique name

def save_mask_index(mask, path="./", index=0):
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to 8-bit grayscale
    # Save the mask image
    file_path = os.path.join(path, f"mask_{index}.png")
    mask_image.save(file_path)  # Save as PNG with a unique name

# Reference from https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L15
def compute_logits_from_mask(mask, eps=1e-3):

    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected mask shape of SAM (256x256)
    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:  # shape matches, do nothing
        pass

    elif logits.shape[0] == logits.shape[1]:  # shape is square
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

    else:  # shape is not square
        # resize the longest side to expected shape
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

        # pad the other side
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits

def build_point_grid_in_mask(n_per_side: int, mask_array: np.ndarray, grids=None) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    if grids is None:
        points = build_point_grid(n_per_side)
    else:
        points = np.array(grids)

    # Convert to pixel coordinates
    height, width = mask_array.shape
    pixel_points = np.array([[int(point[0] * width), int(point[1] * height)] for point in points])

    # Filter out points that fall on the mask
    filtered_pixel_points = []
    for point in pixel_points:
        x, y = point
        if mask_array[y, x] == 255:  # Check if the point is inside the mask
            filtered_pixel_points.append(point)

    return np.array(filtered_pixel_points)

def analyze_line_mask(mask_image, ratio=10):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0
    rect = cv2.minAreaRect(cnt)
    (center), (width, height), angle = rect
    
    # aspect ratio
    min_aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

    # Optional: Calculate solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    # Check if the object is long and thin
    if min_aspect_ratio > ratio:
        return True
    else:
        return False

def sample_grid_from_mask(mask_image, min_area_threshold=10000, grids=None, sample_outside=False):
    binary_mask = check_mask_type(mask_image)

    # connectedComponentsWithStats to separate mask into connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # stats is a list of [x, y, width, height, area] for each component
    input_points_list = []
    input_labels_list = []
    print("num_labels:", num_labels)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        bounding_area = w * h
        # filter out small components based on bounding box area size
        if bounding_area < min_area_threshold:
            continue
        
        # create a mask for each component
        mask = np.zeros_like(binary_mask)
        mask[labels == i] = 255  # Set the component to white

        if (analyze_line_mask(mask)):
            continue
        
        # show mask
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        input_points = build_point_grid_in_mask(128, mask, grids=grids)  # Example usage of build_point_grid_in_mask
        input_labels = np.ones(input_points.shape[0], dtype=int)  # Foreground
        
        # add sample outside mask points
        if sample_outside:
            input_points, input_labels = sample_points_outside_mask(mask, input_points, input_labels)

        if input_points.shape[0] == 0:
            # show mask
            cv2.imshow("mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        # append input points and labels to list
        input_points_list.append(input_points)
        input_labels_list.append(input_labels)
        

    # # Find contours (connected components)
    # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # input_points_list = []
    # input_labels_list = []
    # for cnt in contours:
    #     # measure contour bounding box area
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     area_size = w * h
        
    #     if area_size < min_area_threshold:
    #         continue
    #     # create a mask for each contour
    #     mask = np.zeros_like(binary_mask)
    #     cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the contour

    #     input_points = build_point_grid_in_mask(64, binary_mask, grids=grids)  # Example usage of build_point_grid_in_mask
    #     input_labels = np.ones(input_points.shape[0], dtype=int)  # Foreground
        
    #     for i in range(input_points.shape[0]):
    #         input_one_point = np.array([[input_points[i][0], input_points[i][1]]])
    #         input_one_label = np.array([1])
    #         input_points_list.append(input_one_point)
    #         input_labels_list.append(input_one_label)

    
    return input_points_list, input_labels_list

def contours_to_centroids_sample(contours):
    M = cv2.moments(contours)
    if M["m00"] == 0:
        return None, None
    # Compute centroid from moments
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def test_sample_points_from_mask(mask_image, grids=None, mode='list'):
    binary_mask = check_mask_type(mask_image)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    input_points_list = []
    input_labels_list = []

    for i in range(1, num_labels):
        # create a mask for each component
        mask = np.zeros_like(binary_mask)
        mask[labels == i] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        input_points = build_point_grid_in_mask(64, mask, grids=grids)  # Example usage of build_point_grid_in_mask
        if (input_points.shape[0] == 0):
            cx, cy = contours_to_centroids_sample(cnt)

            input_one_point = np.array([[cx, cy]])
            input_one_label = np.array([1])
            # Sample points outside the mask
            # input_one_point, input_one_label = sample_points_outside_mask(mask_array, input_one_point, input_one_label)
            input_points_list.append(input_one_point)
            input_labels_list.append(input_one_label)
        else:
            input_labels = np.ones(input_points.shape[0], dtype=int)  # Foreground
            
            input_points_list.append(input_points)
            input_labels_list.append(input_labels)

    return input_points_list, input_labels_list

def sample_points_from_mask(mask_image, grids=None, mode='list'):

    binary_mask = check_mask_type(mask_image)

    # Find contours (connected components)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_points_list = []
    input_labels_list = []
    for cnt in contours:
        # create a mask for this component
        mask = np.zeros_like(binary_mask)
        # copy the contour bounding area from binary_mask to mask [contour_width:contour_height]
        x, y, w, h = cv2.boundingRect(cnt)
        mask[y:y+h, x:x+w] = binary_mask[y:y+h, x:x+w]
        input_points = build_point_grid_in_mask(64, mask, grids=grids)  # Example usage of build_point_grid_in_mask
        
        if (input_points.shape[0] == 0):
            cx, cy = contours_to_centroids_sample(cnt)

            input_one_point = np.array([[cx, cy]])
            input_one_label = np.array([1])
            # Sample points outside the mask
            # input_one_point, input_one_label = sample_points_outside_mask(mask_array, input_one_point, input_one_label)
            input_points_list.append(input_one_point)
            input_labels_list.append(input_one_label)
        else:
            input_labels = np.ones(input_points.shape[0], dtype=int)  # Foreground
            
            input_points_list.append(input_points)
            input_labels_list.append(input_labels)

    return input_points_list, input_labels_list

def sample_points_outside_mask(mask_array, input_point, input_label):
    # Define the number of uniform points to sample
    num_uniform_points = 20  

    # Get image dimensions
    height, width = mask_array.shape

    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

    # Create a grid of points uniformly distributed in the image
    uniform_points = build_point_grid(20)  # Example usage of build_point_grid

    # Filter out points that fall on the mask
    filtered_uniform_points = []
    for point in uniform_points:
        x, y = int(point[0] * width), int(point[1] * height)
        if binary_mask[y, x] == 0:  # Check if the point is outside the mask
            filtered_uniform_points.append([x, y])

    filtered_uniform_points = np.array(filtered_uniform_points)

    # Add uniform points to existing input points
    input_point = np.concatenate([input_point, filtered_uniform_points])

    # Add labels for uniform points (label 0)
    input_label = np.concatenate([input_label, np.zeros(len(filtered_uniform_points), dtype=int)]) 
    return input_point, input_label


def load_mask_image_as_sam_input(mask_image):
    binary_mask = check_mask_type(mask_image)

    normalized_mask = (binary_mask / 255.0).astype(np.float32)
    # resize 1x256x256
    normalized_mask = cv2.resize(normalized_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_input = normalized_mask[None, :, :]
    return mask_input

def check_mask_type(mask_image):
    if isinstance(mask_image, str):
        mask_image_path = mask_image
        mask_image = Image.open(mask_image_path).convert("L")  # Convert to grayscale
        mask_array = np.array(mask_image)
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    elif isinstance(mask_image, np.ndarray):
        binary_mask = mask_image
    else:
        raise TypeError("mask_image should be either a file path or a numpy array")

    binary_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
    return binary_mask