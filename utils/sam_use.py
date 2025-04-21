import os
import cv2
import numpy as np
from PIL import Image

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

def sample_points_from_mask(mask_image, mode='list'):

    binary_mask = check_mask_type(mask_image)

    # Find contours (connected components)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_point = []
    input_label = []

    input_points_list = []
    input_labels_list = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # Compute centroid from moments
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            input_point.append([cx, cy])
            input_label.append(1)  # Foreground

            input_one_point = np.array([[cx, cy]])
            input_one_label = np.array([1])
            # Sample points outside the mask
            # input_one_point, input_one_label = sample_points_outside_mask(mask_array, input_one_point, input_one_label)
            input_points_list.append(input_one_point)
            input_labels_list.append(input_one_label)
        else:
            # In case the contour area is zero, skip it
            continue

    input_point = np.array(input_point)
    input_label = np.array(input_label)

    # Sample points outside the mask
    if (mode == 'all'):
        input_point, input_label = sample_points_outside_mask(mask_array, input_point, input_label)
        return input_point, input_label
    elif (mode == 'list'):
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

    return binary_mask