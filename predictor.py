import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import sys
import os
from PIL import Image

from color_filter import ColorFilter

# Put this file in SAM_PROJECT/scripts
#sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sys.path.append("./utils")
import show_view, sam_use, common

def predict_each_separate_area(image_path, mask_path, output_path, max_area_threshold=10000, min_area_threshold=0, usage='default', config_path=None):
    config = None
    grids = sam_use.build_point_grid(128)
    
    # check if config_path is None, if not, load config
    if config_path is not None:
        config = common.load_config(config_path=config_path, field='Predictor')
        print("Predictor use config:", config)
        if usage == 'default':
            max_area_threshold = config['max_area_threshold'][0]
            min_area_threshold = config['min_area_threshold'][0]
        elif usage == 'square':
            max_area_threshold = config['max_area_threshold'][1]
            min_area_threshold = config['min_area_threshold'][1]

    save_path = os.path.join(output_path, usage)
    if (not os.path.exists(save_path)):
        os.makedirs(save_path, exist_ok=True)

    save_mask_path = os.path.join(save_path, 'masks')
    if (not os.path.exists(save_mask_path)):
        os.makedirs(save_mask_path, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = sam_model_setting(config=config)
    predictor.set_image(image)

    # new a black image as mask combine result
    mask_combine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    if (usage == 'default'):
        input_points_list, input_labels_list = sam_use.test_sample_points_from_mask(mask_path, grids=grids, mode='list')
    elif (usage == 'square'):
        input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask_path, min_area_threshold=min_area_threshold, grids=grids, sample_outside=False)

    keep_index_list = []
    for i in range(len(input_points_list)):
        input_point = input_points_list[i]
        input_label = input_labels_list[i]
        
        #mask_input = sam_use.load_mask_image_as_sam_input(mask_path)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            #mask_input=mask_input,
            multimask_output=False,
        )

        sam_use.save_mask_index(masks[0], save_mask_path, index=i)
        # if (i == 22):
        #    show_view.show_mask_with_point(image, masks, scores, input_point, input_label)

        # if mask white area larger or smaller than threshold, skip
        area_size = np.sum(masks[0])
        if ((area_size < min_area_threshold) or (area_size > max_area_threshold)):
            keep_area_flag = False

            # default check area is line or not
            if (usage == 'default'):
                # change from bool to int
                mask_int = masks[0].astype(np.uint8) * 255
                # check if is long line
                if (sam_use.analyze_line_mask(mask_int)):
                    keep_area_flag = True
            if (not keep_area_flag):
                print(f"mask{i} area {area_size} larger or smaller than threshold, skip")
                continue
        keep_index_list.append(i)
        # combine to mask_combine
        mask_combine = np.logical_or(mask_combine, masks[0])
    # save mask_combine as mask_combine.png
    # mask combine to opencv accept format
    mask_combine = mask_combine.astype(np.uint8) * 255
    mask_combine = common.clean_small_area_from_mask(mask_combine, threshold=100)
    mask_combine_image = Image.fromarray(mask_combine)  # Convert to 8-bit grayscale
    mask_combine_image.save(os.path.join(save_path, 'mask_combine.png'))  # Save as PNG with a unique name
    
    # save keep_index_list to file
    keep_index_list_path = os.path.join(save_path, 'keep_index_list.txt')
    with open(keep_index_list_path, 'w') as f:
        for item in keep_index_list:
            f.write("%s\n" % item)
           

def sam_model_setting(config=None):
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    if config is not None:
        sam_checkpoint = config['checkpoint']
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor

def local_test():
    image_path = 'images/mode_scalel_topview.png'
    image_path = r'Ruiguang\9\Ruiguang_9.png'
    #image_path = r'google_earth_test\google_earth.jpg'
    mask_path = 'images/clean_mask.png'
    mask_path = r'Ruiguang\9\clean_mask.png'
    #mask_path = r'google_earth_test\clean_mask_original.png'
    crop_path = './Ruiguang/9' # clean area 15000
    #crop_path = r'google_earth_test\sam_crop' # clean area 8000
    
    predict_each_separate_area(image_path, mask_path, crop_path)
    #predict_from_whole_mask(image_path, mask_path, crop_path)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', '-c', type=str, help='Path to the config file', default=None)
    parser.add_argument('--exist_mask', action='store_true', help='Use exist mask to predict')
    parser.add_argument('--usage', type=str, help='Usage of the script', default='default')
    args = parser.parse_args()
    return args

def set_args_mask_from_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--mask', '-m', type=str, help='Path to the mask image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Folder Path to save the output image', default='./')
    parser.add_argument('--config', '-c', type=str, help='Path to the config file', default=None)
    args = parser.parse_args()
    return args

def main_mask_from_file():
    args = set_args_mask_from_file()
    predict_each_separate_area(args.image, args.mask, args.output, config_path=args.config)

def main():
    args = set_args()
    mask_flag = False

    filter = ColorFilter(args.image, args.output, config=args.config, save_flag=True)

    if args.exist_mask:
        binary_mask = filter.load_existing_mask_binary()
        mask_flag = binary_mask is not None
    
    if not mask_flag:
        binary_mask = filter.filter_color_mask()
        mask_flag = True
    
    if (args.usage == 'default'):
        predict_each_separate_area(args.image, binary_mask, args.output, usage=args.usage, config_path=args.config)
    elif (args.usage == 'square'):
        predict_each_separate_area(args.image, binary_mask, args.output, usage=args.usage, config_path=args.config)

if __name__ == "__main__":
    #local_test()
    main()