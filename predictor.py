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
from sam_mask_selector import SAMMaskSelector
from config import Config

sys.path.append("./mapJson")
from mapJson.mapJsonData import mapJsonData
from mapJson.mapObjData import MapJsonObj

from dataManager.tileManager import TileManager

def predict_process(image, input_points_list, input_labels_list, usage, config, save_mask_path=None, save_line_path=None):
    mask_selector = SAMMaskSelector(config=config.get_all_config())
    predictor = sam_model_setting(config=config)
    predictor.set_image(image)

    # new a black image as mask combine result
    mask_combine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    keep_index_list = []
    keep_area_list = []
    map_obj_list = []
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
        if (save_mask_path is not None):
            sam_use.save_mask_index(masks[0], save_mask_path, index=i)
        # if (i == 22):
        #    show_view.show_mask_with_point(image, masks, scores, input_point, input_label)
        # show_view.show_mask_with_point(image, masks, scores, input_point, input_label)
        # if mask white area larger or smaller than threshold, skip
        area_size = np.sum(masks[0])
        if (not mask_selector.selector(masks[0], i, usage=usage)):
            continue
        
        result_mask = masks[0]
        if (usage == 'line' and save_line_path is not None):
            result_mask = mask_selector.get_selected_mask()
            sam_use.save_mask_index(result_mask, save_line_path, index=i)
        keep_index_list.append(i)
        keep_area_list.append(area_size)
        # combine to mask_combine
        mask_combine = np.logical_or(mask_combine, result_mask)
        
        # # log mask obj list
        # map_json_obj = common.build_map_json_obj(masks[0], i)
        # if map_json_obj is not None:
        #     map_obj_list.append(map_json_obj)

    # save mask_combine as mask_combine.png
    # mask combine to opencv accept format
    mask_combine = mask_combine.astype(np.uint8) * 255
    mask_combine = common.clean_small_area_from_mask(mask_combine, threshold=100)

    return mask_combine, keep_index_list, keep_area_list

def predict_each_sepatate_area_from_image_tile(image_path, mask_path, output_path, max_area_threshold=10000, min_area_threshold=0, usage='default', config_path=None):
    config = None

    config = Config(config_path=config_path)
    pixel_cm = config.get_pixel_cm()

    # create save folder
    save_path = os.path.join(output_path, usage)
    if (not os.path.exists(save_path)):
        os.makedirs(save_path, exist_ok=True)

    save_mask_path = os.path.join(save_path, 'masks')
    if (not os.path.exists(save_mask_path)):
        os.makedirs(save_mask_path, exist_ok=True)

    save_line_path = os.path.join(save_path, 'line_masks')
    if (not os.path.exists(save_line_path)):
        os.makedirs(save_line_path, exist_ok=True)
        
    
    enhance_image_path = os.path.join(os.path.dirname(image_path), "enhanced_image.jpg")
    if os.path.isfile(enhance_image_path):
        image = cv2.imread(enhance_image_path)
    else:
        image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    grids = sam_use.build_point_grid_from_real_size(int(pixel_cm), int(image.shape[1]), int(image.shape[0]))
    if (usage == 'square' or usage == 'yellow'):
        input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask_path, min_area_threshold=min_area_threshold, grids=grids, sample_outside=False)
    else:
        input_points_list, input_labels_list = sam_use.sample_points_from_mask(mask_path, grids=grids)
    
    tileManager = TileManager(
        source_image=image,
        input_points_list=input_points_list,
        input_labels_list=input_labels_list,
        mask_path=mask_path
    )

    tileManager.split_tile()

    image_list, input_points_list_set, input_labels_list_set = tileManager.get_list_of_sam_parameters()

    mask_combine_list = []
    count = 0
    for local_image, local_points_list, local_labels_list in zip(image_list, input_points_list_set, input_labels_list_set):
        # print("local_points_list: ", local_points_list[0])
        # print("local labels list: ", local_labels_list[0])
        # continue

        local_mask_path = os.path.join(save_mask_path, f'{count}')
        if (not os.path.exists(local_mask_path)):
            os.makedirs(local_mask_path, exist_ok=True)

        mask_combine_image, keep_index_list, keep_area_list = predict_process(
            image=local_image,
            input_points_list=local_points_list,
            input_labels_list=local_labels_list,
            usage=usage,
            config=config,
            save_mask_path=local_mask_path
        )
        mask_combine_list.append(mask_combine_image)

        file_name = f"combine_{count}.png"
        cv2.imwrite(os.path.join(save_mask_path, file_name), mask_combine_image)
        sam_use.save_keep_index_list(save_mask_path, keep_index_list, keep_area_list, f"keep_index_list_{count}.txt")
        count = count + 1
    
    result = tileManager.combine_result_from_list(mask_combine_list, save_path)
    mask_combine_image = Image.fromarray(result)  # Convert to 8-bit grayscale
    mask_combine_image.save(os.path.join(save_path, 'tile_combine.png'))  # Save as PNG with a unique name

def predict_each_separate_area(image_path, mask_path, output_path, max_area_threshold=10000, min_area_threshold=0, usage='default', config_path=None):
    config = None
    grids = sam_use.build_point_grid(128)

    config = Config(config_path=config_path)

    # create save folder
    save_path = os.path.join(output_path, usage)
    if (not os.path.exists(save_path)):
        os.makedirs(save_path, exist_ok=True)

    save_mask_path = os.path.join(save_path, 'masks')
    if (not os.path.exists(save_mask_path)):
        os.makedirs(save_mask_path, exist_ok=True)

    save_line_path = os.path.join(save_path, 'line_masks')
    if (not os.path.exists(save_line_path)):
        os.makedirs(save_line_path, exist_ok=True)
        
    
    enhance_image_path = os.path.join(os.path.dirname(image_path), "enhanced_image.jpg")
    if os.path.isfile(enhance_image_path):
        image = cv2.imread(enhance_image_path)
    else:
        image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if (usage == 'square' or usage == 'yellow'):
        input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask_path, min_area_threshold=min_area_threshold, grids=grids, sample_outside=False)
    else:
        input_points_list, input_labels_list = sam_use.sample_points_from_mask(mask_path, grids=grids)

    # test reference: https://github.com/computa`tional-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L115
    # some times more noise
    # input_points_list, input_labels_list = sam_use.compute_points_from_mask(mask_path, min_area_threshold=min_area_threshold)
    # print("local_points_list: ", input_points_list)
    # print("local labels list: ", input_labels_list)
    # return
    mask_combine, keep_index_list, keep_area_list = predict_process(
        image=image,
        input_points_list=input_points_list,
        input_labels_list=input_labels_list,
        save_mask_path=save_mask_path,
        save_line_path=save_line_path,
        usage=usage,
        config=config
    )
    mask_combine_image = Image.fromarray(mask_combine)  # Convert to 8-bit grayscale
    mask_combine_image.save(os.path.join(save_path, 'mask_combine.png'))  # Save as PNG with a unique name
    
    # save keep_index_list to file
    sam_use.save_keep_index_list(save_path, keep_index_list, keep_area_list)
    # bbox = [25.0689, 121.5836, 25.0694, 121.5844]
    # # save to json
    # map_json_data = mapJsonData(
    #     img_name=os.path.basename(image_path).split('.')[0],
    #     bbox_latlon=bbox,
    #     img_size=[image.shape[1], image.shape[0]],
    #     obj_list=map_obj_list
    # )
    # map_json_data.save_to_json(save_path)

def sam_model_setting(config=None):
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    if config is not None:
        sam_checkpoint = config.get('Predictor')['checkpoint']
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
    usage_list = ['default', 'square', 'yellow', 'line']
    usage = args.usage
    if usage not in usage_list:
        print(f"Usage {usage} not in {usage_list}, use default")
        usage = 'default'
    
    mask_flag = False

    filter = ColorFilter(args.image, args.output, config=args.config, save_flag=True)

    if args.exist_mask:

        binary_mask = filter.load_existing_mask_binary(args.usage)
        mask_flag = binary_mask is not None
    
    if not mask_flag:
        binary_mask = filter.filter_color_mask(args.usage)
        mask_flag = True
    
    print("start predict image...")
    #predict_each_separate_area(args.image, binary_mask, args.output, usage=usage, config_path=args.config)
    predict_each_sepatate_area_from_image_tile(args.image, binary_mask, args.output, usage=usage, config_path=args.config)

def run_folder():
    folder = r"D:\GameLab\accident\datas\Xinsheng\part2_ortho"

    folder_list = os.listdir(folder)
    for i in range(len(folder_list)):
        index = i + 1
        image_folder = os.path.join(folder, f"ortho_{index}")
        image_file = os.path.join(image_folder, f"ortho_{index}.tif")
        usage = 'line'

        mask_flag = False
        exist_mask = True
        filter = ColorFilter(image_file, image_folder, config='config.yml', save_flag=True)

        if exist_mask:

            binary_mask = filter.load_existing_mask_binary(usage)
            mask_flag = binary_mask is not None
        
        if not mask_flag:
            binary_mask = filter.filter_color_mask(usage)
            mask_flag = True
        
        predict_each_separate_area(image_file, binary_mask, image_folder, usage=usage, config_path='config.yml')

if __name__ == "__main__":
    #local_test()
    # run_folder()
    main()