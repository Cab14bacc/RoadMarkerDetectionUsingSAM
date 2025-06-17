import os
import cv2
import numpy as np
from PIL import Image
import common

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def build_point_grid_xy(n_per_side_x: int, n_per_side_y: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset_x = 1 / (2 * n_per_side_x)
    offset_y = 1 / (2 * n_per_side_y)
    points_one_side_x = np.linspace(offset_x, 1 - offset_x, n_per_side_x)
    points_one_side_y = np.linspace(offset_y, 1 - offset_y, n_per_side_y)
    grid_x, grid_y = np.meshgrid(points_one_side_x, points_one_side_y)
    points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    return points

def build_point_grid_from_real_size(pixel_cm: int, width: int, height: int, points_interval: int):
    # sample_len = min(width, height)
    n_per_side_x = width * pixel_cm / points_interval
    n_per_side_y = height * pixel_cm / points_interval
    # n_per_side = sample_len * pixel_cm / 32
    return build_point_grid_xy(int(n_per_side_x), int(n_per_side_y))

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

def build_point_grid_in_crop_mask(n_per_side: int, mask_array: np.ndarray, offsets=(0, 0), grids=None, original_size=(1024, 1024)) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    if grids is None:
        points = build_point_grid(n_per_side)
    else:
        points = np.array(grids)

    # Convert to pixel coordinates
    height, width = original_size
    pixel_points = (points * [width, height]).astype(int)
    # Ensure coordinates are within bounds
    pixel_points = np.clip(pixel_points, [0, 0], [width - 1, height - 1])
    
    x_crop = pixel_points[:, 0] - offsets[1]  # subtract x offset
    y_crop = pixel_points[:, 1] - offsets[0]  # subtract y offset

    # Ensure we're indexing inside the crop
    valid_mask = (
        (x_crop >= 0) & (x_crop < mask_array.shape[1]) &
        (y_crop >= 0) & (y_crop < mask_array.shape[0])
    )

    x_crop = x_crop[valid_mask]
    y_crop = y_crop[valid_mask]

    filtered_pixel_points = pixel_points[valid_mask]
    mask_values = mask_array[y_crop, x_crop]
    
    result = filtered_pixel_points[mask_values > 0]
    
    return np.array(result)

def sample_grid_from_mask(mask_image, min_area_threshold=10000, grids=None, sample_outside=False):
    binary_mask = check_mask_type(mask_image)
    original_height, original_width = binary_mask.shape
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
         
        component_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        input_points = build_point_grid_in_crop_mask(128, component_mask, (y, x), grids, original_size=(original_height, original_width))
        
        input_labels = np.ones(input_points.shape[0], dtype=int)  # Foreground

        if input_points.shape[0] == 0:
            continue
            # sample point at the nearest centroid of the mask where mask value is 255 
            cx, cy = find_centroid_in_white(component_mask)
            input_points = np.array([[cx, cy]])
            input_labels = np.array([1])
            

        # append input points and labels to list
        input_points_list.append(input_points)
        input_labels_list.append(input_labels)

    return input_points_list, input_labels_list

# make sure the centroid is on a white mask area
def find_centroid_in_white(mask):

    # mask (np.ndarray): Binary mask (0 and 255).
    # Compute image moments
    M = cv2.moments(mask)

    if M["m00"] == 0:
        # No white area found
        return None

    # Compute centroid coordinates (float)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Check if centroid is already on a white pixel
    if mask[cY, cX] == 255:
        return (cX, cY)

    # Otherwise, find the nearest white pixel
    white_pixels = np.column_stack(np.where(mask == 255))
    # white_pixels: (row, col), need to swap for (x, y)
    dists = np.sum((white_pixels - [cY, cX])**2, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_pixel = white_pixels[nearest_idx]
    nearest_point = (int(nearest_pixel[1]), int(nearest_pixel[0]))  # (x, y)

    return nearest_point[0], nearest_point[1]

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

    return binary_mask

def save_keep_index_list(save_path, keep_index_list, keep_area_list=None, keep_bool_list=None, filename='keep_index_list.txt'):
    # save keep_index_list to file
    keep_index_list_path = os.path.join(save_path, filename)
    with open(keep_index_list_path, 'w') as f:
        for i in range(len(keep_index_list)):
            f.write("%s" % keep_index_list[i])
            if keep_area_list is not None:
                f.write(" %s" % keep_area_list[i])
            if keep_bool_list is not None:
                f.write(f" {keep_bool_list[i]}")
            f.write("\n")
