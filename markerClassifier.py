import cv2
import os
import numpy as np
import argparse
from utils.matchTemplate import match_template_all_rotation
from utils.classifierConfig import ClassifierConfig
from dataManager.mapJsonData import MapJsonData
from utils.common import connected_components_to_scaled_mask

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = ClassifierConfig(config_path)
    base_path = config.get('Classifier')['base_path']
    template_path = config.get('Classifier')['marker_template']
    threshold = config.get('Classifier')['threshold']
    scale_height = config.get('Classifier')['scale_height']
    return base_path, template_path, threshold, scale_height

def padding_mask_region(bbox, component, marker_size=600):
    xs = [pt[0] for pt in component]
    ys = [pt[1] for pt in component]
    
    min_x, min_y, max_x, max_y = bbox

    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1

    # Compute required padding to meet template size
    out_w = max(bbox_w, 600)
    out_h = max(bbox_h, 600)

    # Compute padding on each side to center the component inside the output
    pad_x = (out_w - bbox_w) // 2
    pad_y = (out_h - bbox_h) // 2

    # Allocate output mask
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    xs = np.array(xs)
    ys = np.array(ys)
    # Convert coordinates to relative position in the mask
    rel_xs = xs - min_x + pad_x
    rel_ys = ys - min_y + pad_y
    mask[rel_ys, rel_xs] = 255
    return mask

def match_arrow_template_from_mapjson(args):
    json_path = args.image
    output_path = args.output
    base_path, template_paths, thresholds, scale_height = load_config(args.config)

    map_json_data = MapJsonData.from_file(args.image)
    components = map_json_data.get_components_list()
    bbox_list = map_json_data.get_bbox_list()
    result_components = []
    match_status_list = []
    print("number of components: ", len(components))
    for i, (component, bbox) in enumerate(zip(components, bbox_list)):
        
        mask = padding_mask_region(bbox, component)

        selected = False
        for j in range(len(template_paths)):
            
            template = template_paths[j]
            path = os.path.join(base_path, template)
            threshold = thresholds[j]

            result = match_template_all_rotation(mask, path, args.output,
                                                scale_height=scale_height, threshold=threshold,
                                                pre_name=template, save_flag=False)
            # if there is any white in result
            if np.any(result > 0):
                selected = True
                break
        
        if selected:
            result_components.append(component)
        match_status_list.append(selected)
        print("finished component {}/{}: {}".format(i+1, len(components), selected))
    
    # save match status to txt
    match_status_path = os.path.join(output_path, 'match_status.txt')
    with open(match_status_path, 'w') as f:
        for i, status in enumerate(match_status_list):
            f.write(f"Component {i}: {'Matched' if status else 'Not Matched'}\n")
    # save result components to json
    if result_components:
        shape = map_json_data.original_shape

        result_map_json = MapJsonData(components=result_components, start_coordinate=map_json_data.get_start_coordinate(), original_shape=map_json_data.original_shape)
        result_json_path = os.path.join(output_path, 'marker_result.json')
        result_map_json.save_to_file(result_json_path)
        preview = connected_components_to_scaled_mask(result_components, [shape[0], shape[1]], scaled=0.1)
        # imwrtie mask
        cv2.imwrite(os.path.join(output_path, "marker_result.png"), preview)

def match_arrow_template(args):
    base_path, template_paths, thresholds, scale_height = load_config(args.config)

    result_list = []
    for i in range(len(template_paths)):
        
        template = template_paths[i]
        path = os.path.join(base_path, template)
        print("path: ", path)
        threshold = thresholds[i]

        result = match_template_all_rotation(args.image, path, args.output,
                                             scale_height=scale_height, threshold=threshold,
                                             pre_name=template, save_flag=True)
        result_list.append(result)
    
    # combine all result with or operation
    combine_result = result_list[0]
    for res in result_list[1:]:
        combine_result |= res
    
    filename = os.path.join(args.output, 'result.png')
    cv2.imwrite(filename, combine_result)
    print(f"Result saved to {filename}")

def test():
    # Example usage
    #src_path = 'images/mode_scalel_topview.png'
    args = set_args()
    src_path = 'Ruiguang/7/Ruiguang_7.png'
    src_path = r'D:\GameLab\accident\datas\0004\arrow\tile_combine_fixed.png'
    template_path = 'forward_right_arrow.png'
    output_path = r"D:\GameLab\accident\datas\0004\arrow\classify"
    scale_height = 500
    match_arrow_template(args)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', type=str, help='Path to load the config file', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    match_arrow_template_from_mapjson(args)
