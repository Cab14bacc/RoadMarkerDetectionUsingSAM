import cv2
import numpy as np
import scipy.ndimage
import argparse
import os
import random
import warnings

from scipy.ndimage import label
from skimage.morphology import skeletonize, thin
from colorsys import hsv_to_rgb
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from typing import Optional

from .dataManager.mapJsonData import SplineJsonData, MapJsonData, JsonData 
from .utils.common import connected_components_to_scaled_mask
from .utils.configUtils.splineTestConfig import SplineTestConfig

#region skeleton points function

def cropped_skeleton_coords(region):
    # crop region
    index_y, index_x = np.nonzero(region)
    min_y, max_y = index_y.min(), index_y.max()
    min_x, max_x = index_x.min(), index_x.max()
    cropped_region = region[min_y:max_y+1, min_x:max_x+1]

    # thin function get more smooth result than skeletonize
    skeleton = thin(cropped_region)  # Use thin to get a skeleton
    # skeleton = skeletonize(cropped_region)  # Use thin to get a skeleton
    
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
    x_smooth = scipy.ndimage.gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y_smooth = scipy.ndimage.gaussian_filter1d(y, sigma=sigma, mode="nearest")
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
    x_sorted, y_sorted = smooth_curve(x_sorted, y_sorted, sigma=6)
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

def image_to_spline_data(image_path, debug=False):
    if (isinstance(image_path, np.ndarray)):
        binary_mask = image_path.copy()
    elif os.path.isfile(image_path):
        binary_mask = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        binary_mask = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)        
    # binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (binary_mask > 127).astype(np.uint8)

    smooth_mask = cv2.erode(binary_mask, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    # smooth_mask = gaussian_filter(binary_mask.astype(np.float32), sigma=1)
    # smooth_mask = (smooth_mask > 0.5).astype(np.uint8)

    # save_smooth_mask = smooth_mask * 255
    # cv2.imwrite(os.path.join(output_path, 'smooth_tile_combine.png'), save_smooth_mask)
    # Label connected components
    labeled_mask, num_features = label(binary_mask)
    

    spline_data, skeleton_mask = region_to_spline(labeled_mask, num_features, debug)

    if not debug:
        return spline_data
    else:
        return spline_data, smooth_mask, skeleton_mask

def estimate_linear_tangent(pts, n=10):
    # pts: (N, 2) spline points

    num_pts = len(pts)
    # Ensure n is within bounds
    n = min(n, (num_pts - 1)// 2)

    if n == 0:
        warnings.warn(f"estimate_linear_tangent only works with at least 3 pts, current num of pts: {num_pts}")


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
    tck, u = splprep([x_sorted, y_sorted], s=100)
    # sample 500 points
    u_fine = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_fine, tck)
    spline_pts = np.stack([x_spline, y_spline], axis=1).astype(np.int32)
    
    length = compute_spline_length(spline_pts)
    num_of_samples = int(length / 20)
    u_fine = np.linspace(0, 1, num_of_samples)
    x_spline, y_spline = splev(u_fine, tck)

    x_spline = x_spline.astype(np.int32)
    y_spline = y_spline.astype(np.int32)

    spline_pts = np.stack([x_spline, y_spline], axis=1).astype(np.int32)


    # TODO: set using config
    
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

def is_extension_match(spline_a, spline_b, max_parallel_dist=35, max_endpoint_dist = 1200, max_angle=15, max_projection=10000):
    def check_ray_to_point(ray_origin, ray_dir, target_point):
        # Vector from ray origin to target
        v = target_point - ray_origin
        proj_len = np.dot(v, ray_dir)
        if proj_len < 0 or proj_len > max_projection:
            return False

        # Closest point on ray to target
        closest = ray_origin + proj_len * ray_dir
        dist = np.linalg.norm(closest - target_point)
        return dist < max_parallel_dist
        
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

        # Should be the same as above
        # for sign_a in [1, -1]:
        #     sign_b = 1
        #     dir1 = dir_a * sign_a
        #     dir2 = dir_b * sign_b
        #     angle = angle_between_from_norm(dir1, dir2)
        #     if angle > max_angle:
        #         continue
        #     if check_ray_to_point(origin_a, dir1, origin_b):
        #         return True
            
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

            if np.linalg.norm(origin_a  - origin_b) > max_endpoint_dist:
                continue
            
            if (check_extension_conditions(origin_a, dir_a, origin_b, dir_b=dir_b)):
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


def region_to_spline(labeled_mask, num_features, debug=False):
    spline_data = []
    if debug:
        skeleton_mask = np.zeros((*labeled_mask.shape[:2], 3), np.uint8)


    for i in range(1, num_features + 1):
        region = (labeled_mask == i)

        # crop region and get skeleton coordinates
        coords = cropped_skeleton_coords(region)
        # skip skeleton with not enough points
        if len(coords) < 10:
            continue

        x_sorted, y_sorted = smooth_downsample_skeleton(coords)
        
        if debug:
            for i in range(len(x_sorted) - 1):
                cv2.line(skeleton_mask, (x_sorted[i], y_sorted[i]), (x_sorted[i + 1], y_sorted[i + 1]), (255, 255, 255), thickness=5)


        try:
            spline_pts, x_spline, y_spline = skeleton_to_spline(x_sorted, y_sorted)

            

            spline_length = compute_spline_length(spline_pts)

            # if spline_length < 300:
            #     continue  # Skip short spline
            
            if len(spline_pts) < 3:
                continue  # Skip short spline
            
            # Tangent vectors at endpoints (pointing outwards)
            tan_start, tan_end = estimate_linear_tangent(spline_pts, 10)

            spline_data.append({
                "spline_pts": spline_pts,
                "start": (x_spline[0], y_spline[0]),
                "end": (x_spline[-1], y_spline[-1]),
                "tan_start": tan_start,
                "tan_end": tan_end
            })
            
        except Exception as e:
            print(f"Skipped region {i}: {e}")

    if debug:
        return spline_data, skeleton_mask
    else:
        return spline_data
    
def group_by_direction_and_extension(spline_data, max_parallel_dist=35, max_endpoint_dist=1200, max_projection=1000, condition='both'):
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
                if is_extension_match(spline_data[i], spline_data[j], max_parallel_dist, max_endpoint_dist, max_projection=max_projection) or \
                    is_extension_match(spline_data[j], spline_data[i], max_parallel_dist, max_endpoint_dist, max_projection=max_projection):
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

    # after performing pairwise comparison, we group thoses indirectly adjacent splines
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
    debug_img = result_img.copy()
    cv2.polylines(debug_img, [spline_pts], False, (0,255,0), 3)
    cv2.arrowedLine(debug_img, tuple(spline_pts[0]), 
        tuple((spline_pts[0] + tan_start * max_projection).astype(int)), (255,0,0), 2)
    cv2.arrowedLine(debug_img, tuple(spline_pts[-1]), 
        tuple((spline_pts[-1] + tan_end * max_projection).astype(int)), (0,0,255), 2)
    
    return debug_img
    # result, encoded_img = cv2.imencode(".png", debug_img)
    # encoded_img.tofile('spline_debug_{idx}.png')
    # cv2.imwrite(f'spline_debug_{idx}.png', debug_img)

def debug_each_splines_parallel_detection(spline_data, result_img, max_parallel_dist=10, idx_1=0, idx_2=1):
    spline_pts_1 = spline_data[idx_1]['spline_pts']
    spline_pts_2 = spline_data[idx_2]['spline_pts']

    tan_start_1 = spline_data[idx_1]['tan_start']
    tan_start_2 = spline_data[idx_2]['tan_start']

    tan_end_1 = spline_data[idx_1]['tan_end']
    tan_end_2 = spline_data[idx_2]['tan_end']
    
    points_1 = [np.array(spline_data[idx_1]['start'], dtype=np.int32), np.array(spline_data[idx_1]['end'], dtype=np.int32)]
    points_2 = [np.array(spline_data[idx_2]['start'], dtype=np.int32), np.array(spline_data[idx_2]['end'], dtype=np.int32)]

    tan_1 = [normalize_direction(np.array(spline_data[idx_1]['tan_start'])), 
            normalize_direction(np.array(spline_data[idx_1]['tan_end']))]
    
    tan_2 = [normalize_direction(np.array(spline_data[idx_2]['tan_start'])),
                normalize_direction(np.array(spline_data[idx_2]['tan_end']))]

    debug_img = result_img.copy()

    height, width = debug_img.shape[:2]
    max_projection = int(np.sqrt(height * height + width * width))




    cv2.arrowedLine(debug_img, tuple(spline_pts_1[0]), 
        tuple((spline_pts_1[0] + tan_start_1 * max_projection).astype(int)), (0,127,0), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_1[0]), 
        tuple((spline_pts_1[0] - tan_start_1 * max_projection).astype(int)), (0,64,0), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_1[-1]), 
        tuple((spline_pts_1[-1] + tan_end_1 * max_projection).astype(int)), (0,127,0), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_1[-1]), 
        tuple((spline_pts_1[-1] - tan_end_1 * max_projection).astype(int)), (0,64,0), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_2[0]), 
        tuple((spline_pts_2[0] + tan_start_2 * max_projection).astype(int)), (0,0,127), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_2[0]), 
        tuple((spline_pts_2[0] - tan_start_2 * max_projection).astype(int)), (0,0,64), 2)
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_2[-1]), 
        tuple((spline_pts_2[-1] + tan_end_2 * max_projection).astype(int)), (0,0,127), 2)
    
    
    cv2.arrowedLine(debug_img, tuple(spline_pts_2[-1]), 
        tuple((spline_pts_2[-1] - tan_end_2 * max_projection).astype(int)), (0,0,64), 2)
    
    
    cv2.polylines(debug_img, [spline_pts_1], False, (0,255,0), 3)
    cv2.polylines(debug_img, [spline_pts_2], False, (0,0,255), 3)

    if_grouped = False

    def ray_to_point(ray_origin, ray_dir, target_point, if_grouped):
        # Vector from ray origin to target
        v = target_point - ray_origin
        proj_len = np.dot(v, ray_dir)

        if proj_len < 0 or proj_len > max_projection:
            if_grouped = False or if_grouped
            return np.array([0, 0]), np.array([0, 1]), if_grouped

        # Closest point on ray to target
        closest = ray_origin + proj_len * ray_dir
        dist = np.linalg.norm(closest - target_point)

        if dist < max_parallel_dist:
            if_grouped = True
        
        return closest, target_point, if_grouped
    
    for i in range(2):
        for j in range(2):
            origin_a = points_1[i]
            origin_b = points_2[j]
            dir_a = tan_1[i]
            dir_b = tan_2[j]

            
            for sign_a in [1, -1]:
                for sign_b in [1, -1]:
                    dir1 = dir_a * sign_a
                    dir2 = dir_b * sign_b

                    closest, target_point, if_grouped = ray_to_point(origin_a, dir1, origin_b, if_grouped)
                    ray = closest - target_point 
                    
                    cv2.arrowedLine(debug_img, tuple(target_point.astype(int)), 
                        tuple((closest).astype(int)), (255,255,255), 2)
                    
                    cv2.arrowedLine(debug_img, tuple(target_point.astype(int)), 
                        tuple((target_point + normalize_direction(ray) * max_parallel_dist).astype(int)), (255,0,0), 2)
                    

                    closest, target_point, if_grouped= ray_to_point(origin_b, dir2, origin_a, if_grouped)
                    ray = closest - target_point 

                    cv2.arrowedLine(debug_img, tuple(target_point.astype(int)), 
                        tuple((closest).astype(int)), (255,255,255), 2)
                    
                    cv2.arrowedLine(debug_img, tuple(target_point.astype(int)), 
                        tuple((target_point + normalize_direction(ray) * max_parallel_dist).astype(int)), (255,0,0), 2)

    if if_grouped:
        cv2.putText(debug_img, str("grouped"), np.array([0, 40]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 3, cv2.LINE_AA)

    
    return debug_img
    # result, encoded_img = cv2.imencode(".png", debug_img)
    # encoded_img.tofile('spline_debug_{idx}.png')
    # cv2.imwrite(f'spline_debug_{idx}.png', debug_img)

def temp_compute_average_spline(window_size = 30):
    import matplotlib.pyplot as plt

    
    def point_to_ray_dist(ray_origin, ray_dir, target_point):
        # Vector from ray origin to target
        v = target_point - ray_origin
        proj_len = np.dot(v, ray_dir)

        # Closest point on ray to target
        closest = ray_origin + proj_len * ray_dir
        dist = np.linalg.norm(closest - target_point)
        return dist 


    temp_pts = np.array([
      [4210, 6182],
      [4225, 6228],
      [4240, 6273],
      [4255, 6319],
      [4270, 6365],
      [4285, 6410],
      [4300, 6456],
      [4315, 6501],
      [4330, 6547],
      [4345, 6593],
      [4360, 6638],
      [4375, 6684],
      [4390, 6730],
      [4404, 6775],
      [4419, 6821],
      [4434, 6866],
      [4449, 6912],
      [4464, 6958],
      [4479, 7003],
      [4494, 7049],
      [4509, 7094],
      [4524, 7140],
      [4539, 7186],
      [4554, 7231],
      [4569, 7277],
      [4584, 7322],
      [4599, 7368],
      [4614, 7413],
      [4629, 7459],
      [4644, 7505],
      [4659, 7550],
      [4674, 7596],
      [4184, 6186],
      [4189, 6200],
      [4194, 6215],
      [4199, 6230],
      [4203, 6245],
      [4208, 6260],
      [4213, 6275],
      [4218, 6290],
      [4223, 6305],
      [4228, 6320],
      [4233, 6335],
      [4238, 6350],
      [4243, 6365],
      [4248, 6380],
      [4253, 6395],
      [4258, 6409],
      [4263, 6424],
      [4268, 6439],
      [4273, 6454],
      [4278, 6469],
      [4283, 6484],
      [4288, 6499],
      [4293, 6514],
      [4297, 6529],
      [4302, 6544],
      [4307, 6559],
      [4312, 6574],
      [4317, 6589],
      [4322, 6603],
      [4327, 6618],
      [4331, 6633],
      [4336, 6648],
      [4343, 6665],
      [4348, 6682],
      [4356, 6698],
      [4357, 6712],
      [4363, 6729],
      [4371, 6746],
      [4374, 6764],
      [4378, 6779],
      [4387, 6793],
      [4392, 6811],
      [4394, 6826],
      [4387, 6812],
      [4393, 6827],
      [4401, 6840],
      [4405, 6858],
      [4412, 6874],
      [4418, 6891],
      [4423, 6909],
      [4429, 6926],
      [4435, 6944],
      [4430, 6933],
      [4424, 6916],
      [4417, 6899],
      [4411, 6880],
      [4405, 6862],
      [4400, 6844],
      [4396, 6826],
      [4391, 6808],
      [4386, 6791],
      [4380, 6773],
      [4373, 6755],
      [4363, 6740],
      [4442, 6963],
      [4443, 6969],
      [4445, 6975],
      [4447, 6980],
      [4448, 6986],
      [4450, 6992],
      [4452, 6998],
      [4454, 7003],
      [4456, 7009],
      [4457, 7015],
      [4459, 7020],
      [4461, 7026],
      [4463, 7032],
      [4465, 7038],
      [4467, 7043],
      [4469, 7049],
      [4471, 7055],
      [4472, 7061],
      [4474, 7066],
      [4476, 7072],
      [4478, 7078],
      [4480, 7083],
      [4482, 7089],
      [4484, 7095],
      [4486, 7101],
      [4488, 7106],
      [4490, 7112],
      [4492, 7118],
      [4494, 7123],
      [4496, 7129],
      [4498, 7135],
      [4500, 7140],
      [4508, 7172],
      [4515, 7190],
      [4521, 7208],
      [4526, 7226],
      [4532, 7244],
      [4538, 7263],
      [4543, 7281],
      [4549, 7299],
      [4555, 7317],
      [4561, 7335],
      [4568, 7353],
      [4574, 7371],
      [4580, 7389],
      [4586, 7407],
      [4592, 7426],
      [4598, 7444],
      [4604, 7462],
      [4610, 7480],
      [4616, 7498],
      [4621, 7516],
      [4627, 7535],
      [4633, 7553],
      [4639, 7571],
      [4645, 7589],
      [4651, 7607],
      [4657, 7625],
      [4663, 7643],
      [4669, 7661],
      [4675, 7679],
      [4681, 7698],
      [4686, 7716],
      [4692, 7734],
      [4683, 7626],
      [4688, 7640],
      [4693, 7654],
      [4697, 7668],
      [4702, 7682],
      [4707, 7696],
      [4711, 7711],
      [4716, 7725],
      [4721, 7739],
      [4726, 7753],
      [4730, 7767],
      [4735, 7781],
      [4740, 7795],
      [4744, 7810],
      [4749, 7824],
      [4754, 7838],
      [4758, 7852],
      [4763, 7866],
      [4768, 7880],
      [4773, 7895],
      [4777, 7909],
      [4782, 7923],
      [4787, 7937],
      [4791, 7951],
      [4796, 7965],
      [4801, 7980],
      [4805, 7994],
      [4810, 8008],
      [4814, 8022],
      [4819, 8036],
      [4824, 8051],
      [4828, 8065],
      [4725, 7815],
      [4728, 7824],
      [4730, 7832],
      [4732, 7841],
      [4735, 7849],
      [4737, 7858],
      [4739, 7866],
      [4742, 7875],
      [4744, 7883],
      [4747, 7892],
      [4749, 7900],
      [4751, 7909],
      [4754, 7917],
      [4756, 7926],
      [4759, 7934],
      [4761, 7943],
      [4764, 7951],
      [4767, 7960],
      [4769, 7968],
      [4772, 7977],
      [4775, 7985],
      [4778, 7993],
      [4781, 8002],
      [4784, 8010],
      [4787, 8019],
      [4790, 8027],
      [4793, 8035],
      [4796, 8044],
      [4799, 8052],
      [4803, 8060],
      [4806, 8069],
      [4810, 8077],
      [4840, 8099],
      [4844, 8114],
      [4849, 8128],
      [4854, 8142],
      [4858, 8156],
      [4863, 8171],
      [4868, 8185],
      [4872, 8199],
      [4877, 8213],
      [4882, 8228],
      [4886, 8242],
      [4891, 8256],
      [4896, 8270],
      [4900, 8284],
      [4905, 8299],
      [4910, 8313],
      [4914, 8327],
      [4919, 8341],
      [4924, 8356],
      [4928, 8370],
      [4933, 8384],
      [4938, 8398],
      [4942, 8412],
      [4947, 8427],
      [4952, 8441],
      [4956, 8455],
      [4961, 8469],
      [4966, 8484],
      [4970, 8498],
      [4975, 8512],
      [4980, 8526],
      [4984, 8541],
      [4817, 8106],
      [4831, 8153],
      [4846, 8201],
      [4862, 8248],
      [4878, 8295],
      [4894, 8343],
      [4909, 8390],
      [4924, 8437],
      [4937, 8472],
      [4922, 8509],
      [4941, 8518],
      [4946, 8511],
      [4963, 8557],
      [4980, 8604],
      [4995, 8652],
      [5010, 8699],
      [5026, 8747],
      [5041, 8794],
      [5057, 8841],
      [5073, 8889],
      [5088, 8936],
      [5103, 8984],
      [5087, 8943],
      [5068, 8895],
      [5048, 8848],
      [5029, 8800],
      [5010, 8753],
      [4990, 8706],
      [4970, 8659],
      [4950, 8613],
      [4930, 8566],
      [4905, 8523],
      [4995, 8574],
      [4999, 8586],
      [5003, 8599],
      [5007, 8612],
      [5011, 8625],
      [5015, 8638],
      [5019, 8651],
      [5024, 8664],
      [5028, 8676],
      [5032, 8689],
      [5037, 8702],
      [5041, 8715],
      [5045, 8727],
      [5050, 8740],
      [5054, 8753],
      [5058, 8766],
      [5062, 8779],
      [5066, 8792],
      [5070, 8804],
      [5074, 8817],
      [5079, 8830],
      [5083, 8843],
      [5087, 8855],
      [5092, 8868],
      [5097, 8880],
      [5101, 8893],
      [5105, 8902],
      [5105, 8913],
      [5110, 8913],
      [5103, 8902],
      [5095, 8889],
      [5096, 8885],
      [4813, 8452],
      [4814, 8454],
      [4814, 8457],
      [4815, 8460],
      [4815, 8463],
      [4816, 8465],
      [4817, 8468],
      [4817, 8471],
      [4818, 8474],
      [4818, 8476],
      [4819, 8479],
      [4819, 8482],
      [4820, 8485],
      [4821, 8487],
      [4821, 8490],
      [4822, 8493],
      [4822, 8496],
      [4823, 8498],
      [4824, 8501],
      [4824, 8504],
      [4825, 8507],
      [4826, 8509],
      [4826, 8512],
      [4827, 8515],
      [4828, 8518],
      [4828, 8520],
      [4829, 8523],
      [4830, 8526],
      [4831, 8528],
      [4832, 8531],
      [4832, 8534],
      [4833, 8537]
    ])
    height = 9000
    width = 9000
    temp_pts = temp_pts.astype(np.int32)

    result_spline_pts = []

    for i in range(0,len(temp_pts), window_size // 3):
        pts = [temp_pts[i + j] for j in range(window_size) if i + j < len(temp_pts)]
        if len(pts) < window_size // 2:
            break

        average_pt = np.mean(pts, axis=0)
        result_spline_pts.append(average_pt)
    
    print(len(result_spline_pts))

    result2_spline_pts = []
    for i in range(0,len(result_spline_pts), window_size // 3):
        pts = [result_spline_pts[i + j] for j in range(window_size) if i + j < len(result_spline_pts)]
        if len(pts) < window_size // 2:
            break

        average_pt = np.mean(pts, axis=0)
        result2_spline_pts.append(average_pt)

    result2_spline_pts = np.array(result2_spline_pts)
    
    result3_spline_pts = []
    for pt in temp_pts:
        dist_from_result = []
        for i in range(len(result2_spline_pts) - 1):
            ray = result2_spline_pts[i+1] - result2_spline_pts[i]
            ray = normalize_direction(ray)
            print(result2_spline_pts[i])
            print(ray)
            dist_from_result.append(point_to_ray_dist(result2_spline_pts[i], ray, pt))
        dist_from_result = min(dist_from_result)

        if dist_from_result > 15:
            continue
            
        result3_spline_pts.append(pt) 

    result3_spline_pts = np.array(result3_spline_pts)
    print(result3_spline_pts)

    fig, ax1 = plt.subplots()
    # ax1.set_xlim(0, width)
    # ax1.set_ylim(0, height)

    result_spline_pts = np.array(result_spline_pts)
    ax1.scatter(temp_pts[:, 0], temp_pts[:, 1])
    ax1.plot(result_spline_pts[:, 0], result_spline_pts[:, 1])
    ax1.plot(result2_spline_pts[:, 0], result2_spline_pts[:, 1], 'r-')
    ax1.plot(result3_spline_pts[:, 0], result3_spline_pts[:, 1], 'g--')
    plt.show()



    return result_spline_pts

def draw_splines(spline_data, result_img, colors=[(255, 255, 255)], debug=False):
    result_img = result_img.copy()

    for spline_idx, spline_datum in enumerate(spline_data):
        color = colors[min(spline_idx, len(colors) - 1)]
        # color = tuple(int(c) for c in np.asarray(color).ravel()[:3])
        pts = np.array(spline_datum['spline_pts'], dtype=np.int32)
        
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

def draw_grouped_splines(spline_data, groups, result_img, colors=[(255, 255, 255)], debug=False):
    result_img = result_img.copy()

    for group_id, group in enumerate(groups):
        color = colors[min(group_id, len(colors) - 1)]
        # color = tuple(int(c) for c in np.asarray(color).ravel()[:3])
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

def group_to_mapJsonData(spline_data, groups, output_path, start_coordinate=(0, 0), original_shape=(0, 0)):
    # set compoents list
    components = []
    for group_id, group in enumerate(groups):
        component = []
        for idx in group:
            pts = np.array(spline_data[idx]['spline_pts'], dtype=np.int32)
            for i in range(len(pts)):
                # convert to tuple and add to component
                component.append((pts[i][0], pts[i][1]))
        components.append(component)
    # save to json
    json_data = MapJsonData(components=components, start_coordinate=start_coordinate, original_shape=original_shape)
    # json_data.save_to_file(os.path.join(output_path, "spline_map_data.json"))
    return json_data

def group_to_JsonData(spline_data, groups, original_shape=(0, 0)):
    # set compoents list

    components = []
    for group_id, group in enumerate(groups):
        component = []
        for idx in group:
            pts = np.array(spline_data[idx]['spline_pts'], dtype=np.int32)
            for i in range(len(pts)):
                # convert to tuple and add to component
                component.append((pts[i][0], pts[i][1]))
        components.append(component)

    json_data = JsonData(components=components, original_shape=original_shape)
    # json_data.save_to_file(os.path.join(output_path, "spline_map_data.json"))
    return json_data

def group_to_SplineJsonData(spline_data, groups, road_line_types, left_right_road_line_groups, original_shape=(0, 0)):
    # set compoents list

    # spline_data.append({
    #     "spline_pts": spline_pts, [[x,y], [x,y], ...] 
    #     "start": (x_spline[0], y_spline[0]),
    #     "end": (x_spline[-1], y_spline[-1]),
    #     "tan_start": tan_start,
    #     "tan_end": tan_end
    # })
    
    components = []
    component_pt_spline_indices = []

    for group_idx, group in enumerate(groups):
        component = []
        spline_indices = []
        for local_idx, spline_idx in enumerate(group):
            pts = np.array(spline_data[spline_idx]['spline_pts'], dtype=np.int32)
            for i in range(len(pts)):
                # convert to tuple and add to component
                component.append((pts[i][0], pts[i][1]))
                spline_indices.append(spline_idx)


        components.append(component)
        component_pt_spline_indices.append(spline_indices)


    json_data = SplineJsonData(components, component_pt_spline_indices, 
                                left_right_road_line_groups,
                                road_line_types,
                                original_shape,
                                )
        
    return json_data

def classify_splines(spline_data, groups, image, config: Optional[SplineTestConfig] = None, debug_path = None):

    # load image
    if (isinstance(image, np.ndarray)):
        source_image = image.copy()
    elif os.path.isfile(image):
        source_image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        source_image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # load target color
    white_color = [225, 225, 225]
    yellow_color = [112, 206, 244]
    pixel_cm = 1

    if config is not None:
        white_color = config.get_white_color()
        yellow_color = config.get_yellow_color()
        pixel_cm = config.get_pixel_cm()
        spline_length_threshold = config.get_spline_length_threshold() / pixel_cm
        spline_gap_threshold =  config.get_spline_gap_threshold() / pixel_cm

    # compute extension length
    height, width = source_image.shape[:2]
    max_extension_length = int(np.sqrt(height * height + width * width)) 
    

    if debug_path is not None:
        combined_debug_image = np.zeros_like(source_image)

    road_line_types = []
    new_groups = []
    road_line_groups = []


    for group_idx, group in enumerate(groups):

        #separate left and right road line
        component_spline_data = groups_to_spline_data([group], spline_data)
        local_groups = group_by_direction_and_extension(component_spline_data, 8,
                                                        max_projection=max_extension_length)
        
        # compute arc length of each spline within the current component
        component_spline_arc_lengths = []
        for spline_datum in component_spline_data:
            spline_pts = spline_datum["spline_pts"]
            component_spline_arc_lengths.append(compute_spline_length(spline_pts))

        # select the 2 groups with the largest arc length if more than 2 groups, and remove all else
        if len(local_groups) > 2:
            group_arc_lengths = []

            for local_group in local_groups:
                current_group_arc_length = 0
                for local_idx in local_group:
                    current_group_arc_length += component_spline_arc_lengths[local_idx]

                group_arc_lengths.append(current_group_arc_length)

            sorted_indices = np.argsort(group_arc_lengths)[::-1]
            local_groups = [local_groups[sorted_indices[i]] for i in range(2)]
        
        
        # compute new group and the groups that separate splines into left and right road lines groups
        new_group = []
        left_right_road_line_group = []

        for local_group in local_groups:
            left_right_road_line_group.append([])
            for local_idx in local_group:
                new_group.append(group[local_idx])
                left_right_road_line_group[-1].append(group[local_idx])

        new_groups.append(new_group)
        road_line_groups.append(left_right_road_line_group)



        # sample points from groups to obtain color
        component_colors = []

        for local_group in local_groups:
            current_group_colors = []
            for local_idx in local_group:
                spline_pts = component_spline_data[local_idx]["spline_pts"]
                for spline_pt in spline_pts:
                    x, y = spline_pt
                    color = source_image[y, x]
                    current_group_colors.append(color)
                
            dist_to_white = np.linalg.norm(np.array(current_group_colors) - np.array(white_color), axis=1) # (num of spline pts in component, 3)
            dist_to_white = np.mean(dist_to_white)

            dist_to_yellow = np.linalg.norm(np.array(current_group_colors) - np.array(yellow_color), axis=1)
            dist_to_yellow = np.mean(dist_to_yellow)

            component_color = "yellow" if dist_to_white > dist_to_yellow else "white"
            component_colors.append(component_color) 


        # determine "dashed" or "solid" grouped splines
        # compute the order of the splines within a local group to calculate the average gap
        # we achieve this by computing the nearest endpoint from this endpoint (excluding itself and endpoint from same spline) 
        component_line_styles = []


        for local_group in local_groups:
            if len(local_group) == 1:
                component_line_styles.append("solid")
                continue
                

            spline_endpoints = []

            for local_idx in local_group:
                start = component_spline_data[local_idx]["start"]
                end = component_spline_data[local_idx]["end"]
                spline_endpoints.append(start)
                spline_endpoints.append(end)
            
            spline_endpoints = np.array(spline_endpoints)

            # a NxN array, where [i, j] contains the distance of i endpoint to j endpoint 
            spline_endpoint_dists = np.linalg.norm(spline_endpoints[None, ...] - spline_endpoints[:, None, :], axis=-1)

            spline_arc_lengths = np.array(component_spline_arc_lengths)[local_group]
            median_arc_length = np.median(spline_arc_lengths)

            spline_gaps = []
            for i in range(1, len(spline_endpoint_dists) - 1, 2):
                spline_1_end_idx = i
                spline_2_start_idx = i + 1
                spline_gaps.append(spline_endpoint_dists[spline_1_end_idx][spline_2_start_idx])
                
            median_spline_gap = np.median(spline_gaps)

            # if median arc length is too far away from 400cm
            # or if the gaps between splines is not close to 600
            
            if abs(median_arc_length - (400 / pixel_cm)) > spline_length_threshold or \
                abs(median_spline_gap - (600 / pixel_cm)) > spline_gap_threshold:

                component_line_styles.append("solid")
            else:
                component_line_styles.append("dashed")


        component_type = [" ".join([component_colors[group_idx], component_line_styles[group_idx]]) for group_idx in range(len(local_groups))]

        if debug_path is not None:
            colors = [255 * np.array(hsv_to_rgb(i * (1 / (len(local_groups))), 1, 1)) for i in range(len(local_groups))]
            debug_image = draw_grouped_splines(component_spline_data, local_groups, np.zeros_like(source_image),colors)
            for i, spline_datum in enumerate(component_spline_data):
                debug_pt = spline_datum["start"]
                cv2.putText(debug_image, f"{i}", np.array([debug_pt[0], min(debug_pt[1] + 100 * random.random(), height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)

            for group_idx, local_group in enumerate(local_groups):
                spline = component_spline_data[local_group[0]]
                debug_pt = spline["start"]
                debug_string = " ".join([component_colors[group_idx], component_line_styles[group_idx]])
                cv2.putText(debug_image, debug_string, np.array([debug_pt[0], min(debug_pt[1] + 100* random.random(), height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)


            result, encoded_img = cv2.imencode('.png', debug_image)
            encoded_img.tofile(os.path.join(debug_path, 'debug_classify', f'classify_spline_debug_{group_idx}.png'))        
            combined_debug_image = np.where(np.any(debug_image, -1)[..., None], debug_image, combined_debug_image)
            # debug_image = cv2.resize(debug_image, (0, 0), fx=0.1, fy=0.1)
            # cv2.imshow('spline_debug', debug_image)
            # cv2.waitKey(0)

        road_line_types.append(component_type)

    if debug_path is not None:
        result, encoded_img = cv2.imencode('.png', combined_debug_image)
        encoded_img.tofile(os.path.join(debug_path, f'combined_classify_spline_debug.png'))

    return road_line_types, new_groups, road_line_groups

def filter_splines_by_variance(spline_data, max_variance=0.2):
    filtered_spline_data = []
    spline_variances = compute_spline_variances(spline_data)
    
    for spline_idx, variance in enumerate(spline_variances):
        if variance is not None and variance < max_variance:
            filtered_spline_data.append(spline_data[spline_idx])

    return filtered_spline_data

def compute_spline_variances(spline_data):
    spline_variances = []
    for spline_datum in spline_data:
        segment_vectors = []

        if "spline_pts" not in spline_datum or len(spline_datum["spline_pts"]) <= 1:
            spline_variances.append(None)
            continue

        for spline_pt_idx in range(len(spline_datum["spline_pts"]) - 1):
            pt1 = spline_datum["spline_pts"][spline_pt_idx]
            pt2 = spline_datum["spline_pts"][spline_pt_idx + 1]
            segment_vector = pt2 - pt1
            segment_vectors.append(segment_vector)

        segment_vectors = np.array(segment_vectors)
        segment_angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])

        # 2j because we expect theta and 180 + theta to be the same direction
        resultant = np.mean(np.exp(2j * segment_angles))
        circular_variance = 1 - np.abs(resultant)

        spline_variances.append(circular_variance)
    
    return spline_variances

# def compute_spline_directions(spline_data):
#     """
#     computes spline mean direction with axial symmetry (positive and negative of the same vector is considered the same direction)
#     """
#     mean_directions = []
#     for spline_datum in spline_data:
#         segment_vectors = []

#         if "spline_pts" not in spline_datum or len(spline_datum["spline_pts"]) <= 1:
#             mean_directions.append(None)
#             continue

#         for spline_pt_idx in range(len(spline_datum["spline_pts"]) - 1):
#             pt1 = spline_datum["spline_pts"][spline_pt_idx]
#             pt2 = spline_datum["spline_pts"][spline_pt_idx + 1]
#             segment_vector = pt2 - pt1
#             segment_vectors.append(segment_vector)

#         segment_vectors = np.array(segment_vectors)
#         segment_angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])

#         # 2j because we expect theta and 180 + theta to be the same direction
#         resultant = np.mean(np.exp(2j * segment_angles))
#         mean_angle = 0.5 * np.angle(resultant)
#         mean_dir = np.array([np.cos(mean_angle), np.sin(mean_angle)])
#         mean_directions.append(mean_dir)
    
#     return mean_directions

def keep_main_spline(mask_path, image_path, output_path, config: Optional[SplineTestConfig] = None):
    save_flag = False
    if config is not None:
        save_flag = config.get(field='SplineTest')["save_flag"]

    if config is not None:
        pixel_cm = config.get_pixel_cm()

    spline_data, binary_mask, skeleton_mask = image_to_spline_data(image_path=mask_path, debug=True)
    height, width = binary_mask.shape
    spline_variances = compute_spline_variances(spline_data)

    if save_flag:
        smoothed_mask = binary_mask * 255
        result, encoded_img = cv2.imencode(".png", smoothed_mask)
        encoded_img.tofile(os.path.join(output_path, 'smooth_tile_combine.png'))  
    
        result, encoded_img = cv2.imencode(".png", skeleton_mask)
        encoded_img.tofile(os.path.join(output_path, 'tile_combine_after_skeletonized.png'))  

        result_img = np.zeros((height, width, 3), dtype=np.uint8)
        result_img = draw_splines(spline_data, result_img)
    
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'all_splines.png'))

        for spline_datum, variance in zip(spline_data, spline_variances):
            debug_pt = spline_datum["start"]
            cv2.putText(result_img, f"{variance:.2f}", np.array([debug_pt[0], min(debug_pt[1] + 100, height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3, cv2.LINE_AA)
        
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'all_splines_variance_debug.png'))

        result_img = np.zeros((height, width, 3), dtype=np.uint8)

    if config is not None:
        max_variance = config.get_variance_threshold()
        spline_data = filter_splines_by_variance(spline_data, max_variance)
    else:
        spline_data = filter_splines_by_variance(spline_data)

    # max_extension_length = max(height, width)
    max_extension_length = int(np.sqrt(height * height + width * width))
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length, condition='direction')
    
    if len(groups) == 0:
        return None
    else:
        # find the largest group
        largest_group = [max(groups, key=len)]

   
    if save_flag:
        result_img = draw_grouped_splines(spline_data, groups, result_img)
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'all_splines_after_var_filter.png'))


        for spline_datum in spline_data:
            for spline_pt in spline_datum["spline_pts"]:
                point = spline_datum
                cv2.circle(img=result_img, center=spline_pt, radius=2, color=(255, 0, 0), thickness=2)


        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'all_splines_endpoints.png'))
        result_img = np.zeros((height, width, 3), dtype=np.uint8)


        result_img = draw_grouped_splines(spline_data, largest_group, result_img=result_img)
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'main_spline_result.png'))

    
    # filter_spline_data_by_variance
    spline_data = groups_to_spline_data(largest_group, spline_data)

    # obtain max dists from config
    if config is not None:
        max_parallel_dist = config.get_parallel_dist_threshold()
        max_parallel_dist = max_parallel_dist / pixel_cm if max_parallel_dist is not None else 35

        max_endpoint_dist = config.get_endpoint_dist_threshold()
        max_endpoint_dist = max_endpoint_dist / pixel_cm if max_endpoint_dist is not None else 1200
        groups = group_by_direction_and_extension(spline_data, max_parallel_dist, 
                                                  max_endpoint_dist, max_projection=max_extension_length)
    else:
        groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length)

    # classify each group into a spline type: e.g. [white dashed, white dashed] denotes a double white dashed line
    road_line_types, groups, left_right_road_line_groups = classify_splines(spline_data, groups, image_path, config, debug_path=output_path)

            


    if save_flag:
        # random color for each group
        colors = [255 * np.array(hsv_to_rgb(i * (1 / (len(groups) - 1 + 1e-8)), 1, 1)) for i in range(len(groups))]
        result_img = draw_grouped_splines(spline_data, groups, result_img, colors)
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'main_spline_grouping.png'))
        
        debug_img = result_img.copy()
        for group_id, group in enumerate(groups):
            for idx in group:
                debug_pt = spline_data[idx]['start']
                cv2.putText(debug_img, str(idx), np.array([debug_pt[0], min(debug_pt[1] + 100, height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3, cv2.LINE_AA)
                
        result, encoded_img = cv2.imencode(".png", debug_img)
        encoded_img.tofile(os.path.join(output_path, 'spline_indices.png'))

        # DEBUG
        for group_id, group in enumerate(groups):
            for idx in group:
                debug_pt = spline_data[idx]['start']
                cv2.putText(result_img, str(group_id), np.array([debug_pt[0], min(debug_pt[1] + 100, height)]).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3, cv2.LINE_AA)
                
        result, encoded_img = cv2.imencode(".png", result_img)
        encoded_img.tofile(os.path.join(output_path, 'main_spline_grouping_debug.png'))


        for i in range(len(spline_data)):
            debug_img = debug_each_splines_extension(spline_data, result_img, max_extension_length, i)
            result, encoded_img = cv2.imencode(".png", debug_img)
            encoded_img.tofile(os.path.join(output_path, 'debug_extension', f'spline_debug_{i}.png'))

        for group_idx, group in enumerate(groups):
            for i in range(len(group)):
                for j in range(i, len(group)):
                    idx_1 = group[i]
                    idx_2 = group[j]
                    debug_img = np.zeros_like(result_img)
                    debug_img = debug_each_splines_parallel_detection(spline_data, debug_img, 8, idx_1, idx_2)
                    result, encoded_img = cv2.imencode(".png", debug_img)
                    encoded_img.tofile(os.path.join(output_path, 'debug_parallel', f'spline_debug_{group_idx}_{idx_1}_{idx_2}.png'))
        

    json_data = group_to_SplineJsonData(spline_data, groups, road_line_types, left_right_road_line_groups, (height, width))
    if save_flag:
        json_data.save_to_file(os.path.join(output_path, 'spline_data_result.json'))

    return json_data

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
    
    json_data = group_to_mapJsonData(spline_data, groups, output_path, map_json_data.get_start_coordinate(), map_json_data.get_shape())
    # components = json_data.get_components_list()
    result_img = draw_grouped_splines_scaled(spline_data, groups, json_data.get_shape(), scaled=0.1)
    result, encoded_img = cv2.imencode(".png", result_img)
    encoded_img.tofile(os.path.join(output_path, 'main_spline_grouping.png'))
    # cv2.imwrite(os.path.join(output_path, 'main_spline_grouping.png'), result_img)
    json_data.save_to_file(os.path.join(output_path, "spline_map_data.json"))

def spline_test(image_path, output_path):
    
    spline_data, binary_mask, skeleton_mask= image_to_spline_data(image_path, debug=True)
    height, width = binary_mask.shape
    result_img = np.zeros((height, width, 3), dtype=np.uint8)

    max_extension_length = max(height, width)
    groups = group_by_direction_and_extension(spline_data, max_projection=max_extension_length)
    # random color for each group
    colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(len(groups))]
    result_img = draw_grouped_splines(spline_data, groups, result_img, colors)
    result, encoded_img = cv2.imencode(".png", result_img)
    encoded_img.tofile(os.path.join(output_path, 'spline_result.png'))
    # cv2.imwrite(os.path.join(output_path, 'spline_result.png'), result_img)



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', '-m', type=str, help='Path to the mask file that indicates potential spline areas', required=True)
    parser.add_argument('--image', '-i', type=str, help='Path to the source image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    
    if args.config is not None:
        config = SplineTestConfig(args.config)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    extension_debug_dir_path = os.path.join(args.output, "debug_extension")
    if not os.path.exists(extension_debug_dir_path):
        os.makedirs(extension_debug_dir_path, exist_ok=True)
    
    parallel_debug_dir_path = os.path.join(args.output, "debug_parallel")
    if not os.path.exists(parallel_debug_dir_path):
        os.makedirs(parallel_debug_dir_path, exist_ok=True)
    
    classify_debug_dir_path = os.path.join(args.output, "debug_classify")
    if not os.path.exists(classify_debug_dir_path):
        os.makedirs(classify_debug_dir_path, exist_ok=True)

    # temp_compute_average_spline()
    #spline_test(args.image, args.output)
    # check stem .png
    if os.path.splitext(args.mask)[1] in ['.png', '.jpg']:
        if args.config is not None:
            keep_main_spline(args.mask, args.image, args.output, config)
        else:
            keep_main_spline(args.mask, args.image, args.output)
                
    elif os.path.splitext(args.image)[1] == '.json':
        mapJson_to_spline(args.image, args.output)