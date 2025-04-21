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

def predict_from_whole_mask(image_path, mask_path, output_path, config=None):
    crop_path = os.path.join(output_path, 'sam_crop')
    if (not os.path.exists(crop_path)):
        os.makedirs(crop_path, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = sam_model_setting()
    predictor.set_image(image)

    # use whole mask as sample reference
    input_point, input_label = sam_use.sample_points_from_mask(mask_path, mode='all')
    print("Input Points:", input_point)
    print("Input Labels:", input_label)

    mask_input = sam_use.load_mask_image_as_sam_input(mask_path)
    print("Mask Input Shape:", mask_input.shape)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input,
        multimask_output=False,
    )

    print("logits shape:", logits.shape)
    sam_use.save_mask(masks, crop_path)
    show_view.show_mask_with_point(image, masks, scores, input_point, input_label)

def predict_each_separate_area(image_path, mask_path, output_path, area_threshold=10000, config_path=None):
    config = None
    # check if config_path is None, if not, load config
    if config_path is not None:
        config = common.load_config(config_path=config_path, field='Predictor')
        print("Predictor use config:", config)
        area_threshold = config['area_threshold']

    crop_path = os.path.join(output_path, 'sam_crop')
    if (not os.path.exists(crop_path)):
        os.makedirs(crop_path, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = sam_model_setting(config=config)
    predictor.set_image(image)

    # new a black image as mask combine result
    mask_combine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    input_points_list, input_labels_list = sam_use.sample_points_from_mask(mask_path, mode='list')
    for i in range(len(input_points_list)):
        input_point = input_points_list[i]
        input_label = input_labels_list[i]
        
        mask_input = sam_use.load_mask_image_as_sam_input(mask_path)
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input,
            multimask_output=False,
        )

        # create and name folder as i 
        folder_name = f"mask_{i}"
        save_mask_path = os.path.join(crop_path, folder_name)
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path, exist_ok=True)

        sam_use.save_mask(masks, save_mask_path)
        #if (i == 52):
        #    show_mask_with_point(image, masks, scores, input_point, input_label)
        # if mask white area larger than threshold, skip
        area_size = np.sum(masks[0])
        if ((area_size > area_threshold)):
            print(f"mask{i} area {area_size} larger than threshold, skip")
            continue

        # combine to mask_combine
        mask_combine = np.logical_or(mask_combine, masks[0])
    # save mask_combine as mask_combine.png
    # mask combine to opencv accept format
    mask_combine = mask_combine.astype(np.uint8) * 255
    mask_combine = common.clean_small_area_from_mask(mask_combine, threshold=100)
    mask_combine_image = Image.fromarray(mask_combine)  # Convert to 8-bit grayscale
    mask_combine_image.save(os.path.join(crop_path, 'mask_combine.png'))  # Save as PNG with a unique name
        
           

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
    args = parser.parse_args()
    return args

def set_args_mask_from_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--mask', '-m', type=str, help='Path to the mask image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Folder Path to save the output image', default='./')
    args = parser.parse_args()
    return args

def main_mask_from_file():
    args = set_args_mask_from_file()
    predict_each_separate_area(args.image, args.mask, args.output)
    #predict_from_whole_mask(args.image, args.mask, args.output)

def main():
    args = set_args()

    filter = ColorFilter(args.image, args.output, config=args.config, save_flag=True)
    binary_mask = filter.filter_color_mask()
    predict_each_separate_area(args.image, binary_mask, args.output, config_path=args.config)

if __name__ == "__main__":
    #local_test()
    #main_mask_from_file()
    main()