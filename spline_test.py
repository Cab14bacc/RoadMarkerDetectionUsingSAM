import cv2
import numpy as np
from scipy.ndimage import label
import scipy.ndimage
from skimage.morphology import skeletonize, thin
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

import argparse
import os
import random

from dataManager.mapJsonData import MapJsonData
from utils.common import connected_components_to_scaled_mask

#region skeleton points function

def cropped_skeleton_coords(region):
    # crop region
    index_y, index_x = np.nonzero(region)
    min_y, max_y = index_y.min(), index_y.max()
    min_x, max_x = index_x.min(), index_x.max()
    cropped_region = region[min_y:max_y+1, min_x:max_x+1]

    # thin function get more smooth result than skeletonize
    skeleton = thin(cropped_region)  # Use thin to get a skeleton
    
    coords = np.column_stack(np.nonzero(skeleton))

    # add back the offset
    coords[:, 0] += min_y
    coords[:, 1] += min_x

    return coords
# sort line order to fit continuous line
def sort_skeleton_points(x, y):
    coords = np.stack([x, y], axis=1)
    tree = cKDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    path = [0]
    visited[0] = True
    for _ in range(1, len(coords)):
        dists, indices = tree.query(coords[path[-1]], k=len(coords))
        for i in indices:
            if not visited[i]:
                path.append(i)
                visited[i] = True
                break
    coords_sorted = coords[path]
    return coords_sorted[:, 0], coords_sorted[:, 1]
def smooth_curve(x, y, sigma=2):
    x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=sigma)
    y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=sigma)
    return x_smooth, y_smooth
def remove_duplicates(x, y, eps=1e-3):
    coords = np.stack([x, y], axis=1)
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    keep = np.insert(dists > eps, 0, True)
    return x[keep], y[keep]
def downsample(x, y, step=2):
    return x[::step], y[::step]

def smooth_downsample_skeleton(coords):
    y, x = coords[:, 0], coords[:, 1]
    x_sorted, y_sorted = sort_skeleton_points(x, y)
    x_sorted, y_sorted = smooth_curve(x_sorted, y_sorted, sigma=2)
    x_sorted, y_sorted = downsample(x_sorted, y_sorted, step=2)
    if len(x_sorted) > 6:
        x_sorted = x_sorted[2:-2]
        y_sorted = y_sorted[2:-2]
    x_sorted, y_sorted = remove_duplicates(x_sorted, y_sorted)
    return x_sorted, y_sorted

#endregion

#region spline data functions

def mapJson_to_spline_data(map_json_data):
    components = map_json_data.get_components_list()
    bbox_list = map_json_data.get_bbox_list()
    shape = map_json_data.get_shape()
    spline_data = []

    for i, (component, bbox) in enumerate(zip(components, bbox_list)):
        # bbox to shape
        bbox_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        # create bbox mask
        mask = np.zeros((bbox_shape[1], bbox_shape[0]), dtype=np.uint8)

        xs = np.array([pt[0] for pt in component])
        ys = np.array([pt[1] for pt in component])
        xs = xs - bbox[0]
        ys = ys - bbox[1]
        
        # set max of xs and ys clip to mask shape
        xs = np.clip(xs, 0, bbox_shape[0] - 1)
        ys = np.clip(ys, 0, bbox_shape[1] - 1)
        
        mask[ys, xs] = 255
        # get skeleton coordinates
        coords = cropped_skeleton_coords(mask)
        if len(coords) < 10:
            continue

        x_sorted, y_sorted = smooth_downsample_skeleton(coords)

        try:
            spline_pts, x_spline, y_spline = skeleton_to_spline(x_sorted, y_sorted)

            spline_length = compute_spline_length(spline_pts)
            if spline_length < 300:
                continue  # Skip short spline
            # Tangent vectors at endpoints
            tan_start, tan_end = estimate_linear_tangent(spline_pts, 5)

            spline_data.append({
                "spline_pts": spline_pts,
                "start": (x_spline[0], y_spline[0]),
                "end": (x_spline[-1], y_spline[-1]),
                "tan_start": tan_start,
                "tan_end": tan_end
            })
            
        except Exception as e:
            print(f"Skipped region {i}: {e}")
    return spline_data

def groups_to_spline_data(groups, spline_data):
    result_spline_data = []
    for group in groups:
        for idx in group:
            result_spline_data.append(spline_data[idx])
    return result_spline_data

def image_to_spline_data(image_path):
    binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (binary_mask > 127).astype(np.uint8)
    
    smooth_mask = gaussian_filter(binary_mask.astype(np.float32), sigma=1)
    smooth_mask = (smooth_mask > 0.5).astype(np.uint8)
    # save_smooth_mask = smooth_mask * 255
    # cv2.imwrite(os.path.join(output_path, 'smooth_tile_combine.png'), save_smooth_mask)
    # Label connected components
    labeled_mask, num_features = label(smooth_mask)

    spline_data = region_to_spline(labeled_mask, num_features)
    return spline_data, smooth_mask

def estimate_linear_tangent(pts, n=10):
    # pts: (N, 2) spline points
    if len(pts) < n + 1:
        n = len(pts) - 1

    dx_start = pts[n][0] - pts[n+n][0]
    dy_start = pts[n][1] - pts[n+n][1]
    tan_start = np.array([dx_start, dy_start])
    tan_start = tan_start / (np.linalg.norm(tan_start) + 1e-8)

    dx_end = pts[-n][0] - pts[-n-n][0]
    dy_end = pts[-n][1] - pts[-n-n][1]
    tan_end = np.array([dx_end, dy_end])
    tan_end = tan_end / (np.linalg.norm(tan_end) + 1e-8)

    return tan_start, tan_end

def compute_spline_length(spline_pts):
    diffs = np.diff(spline_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

def skeleton_to_spline(x_sorted, y_sorted):
    tck, u = splprep([x_sorted, y_sorted], s=20)
    # sample 500 points
    u_fine = np.linspace(0, 1, 32)

    x_spline, y_spline = splev(u_fine, tck)
    spline_pts = np.stack([x_spline, y_spline], axis=1).astype(np.int32)
    return spline_pts, x_spline, y_spline

#endregion

#region extension functions

def normalize_direction(v):
    return v / (np.linalg.norm(v) + 1e-8)
def angle_between(v1, v2):
    v1 = normalize_direction(v1)
    v2 = normalize_direction(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def angle_between_from_norm(v1, v2):
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def is_extension_match(spline_a, spline_b, max_dist=50, max_angle=15, max_projection=1000):
    def check_ray_to_point(ray_origin, ray_dir, target_point):
        # Vector from ray origin to target
        v = target_point - ray_origin
        proj_len = np.dot(v, ray_dir)
        if proj_len < 0 or proj_len > max_projection:
            return False

        # Closest point on ray to target
        closest = ray_origin + proj_len * ray_dir
        dist = np.linalg.norm(closest - target_point)
        return dist < max_dist
        
    def check_extension_conditions(origin_a, dir_a, origin_b, dir_b):
        for sign_a in [1, -1]:
            for sign_b in [1, -1]:
                dir1 = dir_a * sign_a
                dir2 = dir_b * sign_b
                angle = angle_between_from_norm(dir1, dir2)
                if angle > max_angle:
                    continue
                if check_ray_to_point(origin_a, dir1, origin_b):
                    return True
        return False
    
    points_a = [np.array(spline_a['start']), np.array(spline_a['end'])]
    points_b = [np.array(spline_b['start']), np.array(spline_b['end'])]

    tan_a = [normalize_direction(np.array(spline_a['tan_start'])), 
                normalize_direction(np.array(spline_a['tan_end']))]
    
    tan_b = [normalize_direction(np.array(spline_b['tan_start'])),
                normalize_direction(np.array(spline_b['tan_end']))]
    
    # use all directions and check extension conditions
    for i in range(2):
        for j in range(2):
            origin_a = points_a[i]
            origin_b = points_b[j]
            dir_a = tan_a[i]
            dir_b = tan_b[j]
            if (check_extension_conditions(origin_a, dir_a, origin_b, dir_b)):
                return True
    
    return False

def is_direction_match(spline_a, spline_b, max_angle=15):

    tan_a = [normalize_direction(np.array(spline_a['tan_start'])), 
                normalize_direction(np.array(spline_a['tan_end']))]
    
    tan_b = [normalize_direction(np.array(spline_b['tan_start'])),
                normalize_direction(np.array(spline_b['tan_end']))]
    
    for i in range(2):
        for j in range(2):
            dir_a = tan_a[i]
            dir_b = tan_b[j]
            if (compare_spline_direction_similarity(dir_a, dir_b, max_angle)):
                return True
    return False

def compare_spline_direction_similarity(dir_a, dir_b, max_angle=15):
    for sign_a in [1, -1]:
        for sign_b in [1, -1]:
            dir1 = dir_a * sign_a
            dir2 = dir_b * sign_b
            angle = angle_between_from_norm(dir1, dir2)
            if angle < max_angle:
                return True
    return False

#endregion

def region_to_spline(labeled_mask, num_features):
    spline_data = []
    
    for i in range(1, num_features + 1):
        region = (labeled_mask == i)

        # crop region and get skeleton coordinates
        coords = cropped_skeleton_coords(region)
        # skip skeleton with not enough points
        if len(coords) < 10:
            continue

        x_sorted, y_sorted = smooth_downsample_skeleton(coords)

        try:
            spline_pts, x_spline, y_spline = skeleton_to_spline(x_sorted, y_sorted)

            spline_length = compute_spline_length(spline_pts)
            if spline_length < 300:
                continue  # Skip short spline
            # Tangent vectors at endpoints
            tan_start, tan_end = estimate_linear_tangent(spline_pts, 5)

            spline_data.append({
                "spline_pts": spline_pts,
                "start": (x_spline[0], y_spline[0]),
                "end": (x_spline[-1], y_spline[-1]),
                "tan_start": tan_start,
                "tan_end": tan_end
            })
            
        except Exception as e:
            print(f"Skipped region {i}: {e}")

    return spline_data

def group_by_direction_and_extension(spline_data, max_projection=1000, condition='both'):
    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for j in adj[i]:
            if not visited[j]:
                dfs(j, group)

    n = len(spline_data)
    adj = [[] for _ in range(n)]

    # check all pairs of splines for extension match
    for i in range(n):
        for j in range(i + 1, n):
            if (condition == 'both'):
                if is_extension_match(spline_data[i], spline_data[j], max_projection=max_projection) or is_extension_match(spline_data[j], spline_data[i], max_projection=max_projection):
                    # print("spline {} and {} are extensions".format(i, j))
                    adj[i].append(j)
                    adj[j].append(i)
            elif (condition == 'direction'):
                if is_direction_match(spline_data[i], spline_data[j], 15):
                    # print("spline {} and {} are same directions".format(i, j))
                    adj[i].append(j)
                    adj[j].append(i)
            

    visited = [False] * n
    groups = []

    # find all pairs relative and group them
    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    return groups

def debug_each_splines_extension(spline_data, result_img, max_projection=1000, idx=0):
    spline_pts = spline_data[idx]['spline_pts']
    tan_start = spline_data[idx]['tan_start']
    tan_end = spline_data[idx]['tan_end']
    debug_img = np.zeros_like(result_img)
    cv2.polylines(debug_img, [spline_pts], False, (0,255,0), 3)
    cv2.arrowedLine(debug_img, tuple(spline_pts[0]), 
        tuple((spline_pts[0] + tan_start * max_projection).astype(int)), (255,0,0), 2)
    cv2.arrowedLine(debug_img, tuple(spline_pts[-1]), 
        tuple((spline_pts[-1] + tan_end * max_projection).astype(int)), (0,0,255), 2)
    cv2.imwrite(f'spline_debug_{idx}.png', debug_img)

def draw_grouped_splines(spline_data, groups, result_img, colors=[(255, 255, 255)], debug=False):
    for group_id, group in enumerate(groups):
        color = colors[min(group_id, len(colors) - 1)]
        for idx in group:
            pts = np.array(spline_data[idx]['spline_pts'], dtype=np.int32)
            
            # draw spline on result image
            for i in range(len(pts) - 1):
                cv2.line(result_img, tuple(pts[i]), tuple(pts[i+1]), color, thickness=5)
            if debug:
                # scale down image and show
                debug_image = result_img.copy()
                debug_image = cv2.resize(debug_image, (0, 0), fx=0.5, fy=0.5)

                cv2.imshow('spline_debug', debug_image)
                cv2.waitKey(0)
    return result_img

def draw_grouped_splines_scaled(spline_data, groups, shape, colors=[(255, 255, 255)], scaled=0.1):
    height, width = shape
    scaled_shape = (int(height * scaled), int(width * scaled))
    result_img = np.zeros((scaled_shape[0], scaled_shape[1], 3), dtype=np.uint8)
    for group_id, group in enumerate(groups):
        color = colors[min(group_id, len(colors) - 1)]
        for idx in group:
            pts = np.array(spline_data[idx]['spline_pts'], dtype=np.int32)
            # draw spline on result image
            for i in range(len(pts) - 1):
                x = pts[i][0] / width * scaled_shape[1]
                y = pts[i][1] / height * scaled_shape[0]
                x_next = pts[i+1][0] / width * scaled_shape[1]
                y_next = pts[i+1][1] / height * scaled_shape[0]
                cv2.line(result_img, (int(y), int(x)), (int(y_next), int(x_next)), color, thickness=5)
                # cv2.line(result_img, tuple(pts[i]), tuple(pts[i+1]), color, thickness=5)
    return result_img

def group_to_mapJson(spline_data, groups, output_path, start_coordinate=(0, 0), original_shape=(0, 0)):
    # set compoents list
    components = []
    for group_id, group in enumerate(groups):
        component = []
        for idx in group:
            pts = np.array(spline_data[idx]['spline_pts'], dtype=np.int32)
            for i in range(len(pts) - 1):
                # convert to tuple and add to component
                component.append((pts[i][0], pts[i][1]))
        components.append(component)
    # save to json
    json_data = MapJsonData(components=components, start_coordinate=start_coordinate, original_shape=original_shape)
    # json_data.save_to_file(os.path.join(output_path, "spline_map_data.json"))
    return json_data

def keep_main_spline(image_path, output_path):

    spline_data, binary_mask = image_to_spline_data(image_path)
    height, width = binary_mask.shape
    result_img = np.zeros((height, width, 3), dtype=np.uint8)

    max_extension_length = max(height, width)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length, condition='direction')
    # find the largest group
    largest_group = [max(groups, key=len)]
    result_img = draw_grouped_splines(spline_data, largest_group, result_img)
    cv2.imwrite(os.path.join(output_path, 'main_spline_result.png'), result_img)

    spline_data = groups_to_spline_data(largest_group, spline_data)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length)
    # random color for each group
    colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(len(groups))]
    result_img = draw_grouped_splines(spline_data, groups, result_img, colors)
    cv2.imwrite(os.path.join(output_path, 'main_spline_grouping.png'), result_img)

def mapJson_to_spline(map_json_path, output_path):
    map_json_data = MapJsonData.from_file(map_json_path)
    spline_data = mapJson_to_spline_data(map_json_data)

    height, width = map_json_data.get_shape()[:2]
    max_extension_length = max(height, width)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length, condition='direction')
    # find the largest group
    largest_group = [max(groups, key=len)]
    spline_data = groups_to_spline_data(largest_group, spline_data)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length)
    
    json_data = group_to_mapJson(spline_data, groups, output_path, map_json_data.get_start_coordinate(), map_json_data.get_shape())
    # components = json_data.get_components_list()
    result_img = draw_grouped_splines_scaled(spline_data, groups, json_data.get_shape(), scaled=0.1)
    cv2.imwrite(os.path.join(output_path, 'main_spline_grouping.png'), result_img)
    json_data.save_to_file(os.path.join(output_path, "spline_map_data.json"))

def spline_test(image_path, output_path):
    
    spline_data, binary_mask = image_to_spline_data(image_path)
    height, width = binary_mask.shape
    result_img = np.zeros((height, width, 3), dtype=np.uint8)

    max_extension_length = max(height, width)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length)
    # random color for each group
    colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(len(groups))]
    result_img = draw_grouped_splines(spline_data, groups, result_img, colors)
    cv2.imwrite(os.path.join(output_path, 'spline_result.png'), result_img)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    #spline_test(args.image, args.output)
    # check stem .png
    if os.path.splitext(args.image)[1] == '.png':
        keep_main_spline(args.image, args.output)
    elif os.path.splitext(args.image)[1] == '.json':
        mapJson_to_spline(args.image, args.output)