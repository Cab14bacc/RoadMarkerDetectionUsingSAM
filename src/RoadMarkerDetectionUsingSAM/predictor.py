import numpy as np
import argparse
import cv2
import os
import time
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

from .color_filter import ColorFilter
from .utils import show_view, sam_use, common
from .utils.sam_mask_selector import SAMMaskSelector
from .utils.configUtils.predictorConfig import PredictorConfig
from .utils.tiffLoader import TiffLoader
from .dataManager.tileManager import SamTileManager
from .dataManager.bigTiffManager import BigTiffManager
from .dataManager.mapJsonData import MapJsonData

def predict_process(image, predictor, input_points_list, input_labels_list, usage, config, save_mask_path=None, save_line_path=None, index=None):
    mask_selector = SAMMaskSelector(config=config)
    save_flag = False
    if config is not None:
        save_flag = config.get(field='Predictor')['save_flag']
    # new a black image as mask combine result
    mask_combine_after_seg = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask_combine_after_selector = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    keep_index_list = []
    keep_area_list = []
    keep_bool_list = []
    map_obj_list = []
    
    mask_count = 0

    step_size = 4

    for i in range(len(input_points_list)):

        # input_point = input_points_list[i]
        # input_label = input_labels_list[i]

        input_points = input_points_list[i]
        input_labels = input_labels_list[i]

        list_index = list(range(len(input_points_list[i])))
        np.random.shuffle(list_index)

        input_points = np.array(input_points)[list_index].tolist()
        input_labels = np.array(input_labels)[list_index].tolist()
        
        for j in range(0,len(input_points), step_size):
            if (len(input_points) - j) < (step_size * 0.75):
                continue

            if j + step_size - 1 >= len(input_points):
                point_coords = np.array([input_points[k] for k in range(j, len(input_points), 1)])
                point_labels = np.array([input_labels[k] for k in range(j, len(input_points), 1)])

            else:
                point_coords = np.array([input_points[j + k] for k in range(step_size)])
                point_labels = np.array([input_labels[j + k] for k in range(step_size)])



            # mask_input = sam_use.load_mask_image_as_sam_input(mask_path)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                #mask_input=mask_input,
                multimask_output=True,
            )
            max_idx = np.argmax(scores)

            scores = [scores[max_idx]]
            masks = [masks[max_idx]]

            # if (i == 145):
            #     show_view.show_mask_with_point(image, masks, scores, input_point, input_label)
            # if mask white area larger or smaller than threshold, skip
            # show_view.show_mask_with_point(image, masks, scores, point_coords, point_labels)

            area_size = np.sum(masks[0])
            keep_index_list.append(i)
            keep_area_list.append(area_size)
            result_mask = masks[0]

            if save_flag and (save_mask_path is not None):
                # sam_use.save_mask_index(result_mask, save_mask_path, index=i)
                result_mask_with_sam_input = show_view.show_mask_with_point_cv2(image, masks, scores, point_coords, point_labels, True)[0]
                result_mask_with_sam_input = cv2.cvtColor(result_mask_with_sam_input, cv2.COLOR_RGB2BGR)
                result, encoded_img = cv2.imencode(".png", result_mask_with_sam_input)
                encoded_img.tofile(os.path.join(save_mask_path, f"mask_{mask_count}.png"))
                # sam_use.save_mask_index(result_mask_with_sam_input, save_mask_path, index=mask_count)
                mask_count += 1
            
            mask_combine_after_seg = np.logical_or(mask_combine_after_seg, result_mask)

            if (not mask_selector.selector(masks[0], i, usage=usage)):
                keep_bool_list.append(False)
                continue
            
            keep_bool_list.append(True)
            result_mask = mask_selector.get_selected_mask()
            if (usage == 'line'):
                if save_flag and save_line_path is not None:
                    sam_use.save_mask_index(result_mask, save_line_path, index=i)

            # combine to mask_combine
            mask_combine_after_selector = np.logical_or(mask_combine_after_selector, result_mask)
        
    # save mask_combine as mask_combine.png
    # mask combine to opencv accept format
    mask_combine_after_seg = mask_combine_after_seg.astype(np.uint8) * 255
    mask_combine_after_selector = mask_combine_after_selector.astype(np.uint8) * 255
    #mask_combine = common.clean_small_area_from_mask(mask_combine, threshold=100)

    return mask_combine_after_seg, mask_combine_after_selector, keep_index_list, keep_area_list, keep_bool_list

def predict_each_sepatate_area_from_image_tile(image_path, mask_path, output_path, negative_mask = None, max_area_threshold=10000, min_area_threshold=0, usage='default', config=None):

    save_flag = False
    if config is not None:
        save_flag = config.get(field='Predictor')['save_flag']
    pixel_cm = config.get_pixel_cm()

    # create save folder, and clears it if it exists
    save_path = None
    enhance_image_path = None
    if save_flag:
        save_path = os.path.join(output_path, usage)
        if (not os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        else:
            for item in os.listdir(save_path):
                target_file = os.path.join(save_path, item)
                if os.path.isfile(target_file):
                    os.remove(target_file)

        save_mask_path = os.path.join(save_path, 'masks')
        if (not os.path.exists(save_mask_path)):
            os.makedirs(save_mask_path, exist_ok=True)
        else:
            for item in os.listdir(save_mask_path):
                target_file = os.path.join(save_mask_path, item)
                if os.path.isfile(target_file):
                    os.remove(target_file)

        save_line_path = os.path.join(save_path, 'line_masks')
        if (not os.path.exists(save_line_path)):
            os.makedirs(save_line_path, exist_ok=True)
        else:
            for item in os.listdir(save_line_path):
                target_file = os.path.join(save_line_path, item)
                if os.path.isfile(target_file):
                    os.remove(target_file)
        
        enhance_image_path = os.path.join(os.path.dirname(image_path), "enhanced_image.jpg")

    if (isinstance(image_path, np.ndarray)):
        image = image_path
    elif os.path.isfile(enhance_image_path):
        image = cv2.imdecode(np.fromfile(enhance_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.imread(enhance_image_path)
    else:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start_sample_time = time.perf_counter()
    sample_points_interval = config.get_sample_points_interval(usage)
    grids = sam_use.build_point_grid_from_real_size(int(pixel_cm), int(image.shape[1]), int(image.shape[0]), int(sample_points_interval))
    
    if negative_mask is not None and os.path.isfile(negative_mask):
        negative_mask = cv2.imdecode(np.fromfile(negative_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if os.path.isfile(mask_path):
            mask_path = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        negative_mask = np.logical_not(negative_mask)
        mask_path = np.logical_and(mask_path, negative_mask).astype(np.uint8) * 255
        negative_mask = negative_mask.astype(np.uint8) * 255
        
        
        # temp = negative_mask.astype(np.uint8) * 255
        # new_shape = (temp.shape[1] // 4, temp.shape[0] // 4)  # (width, height)
        # resized = cv2.resize(temp, new_shape)
        # cv2.imshow("sdlkfj", resized)

        
        # new_shape = (mask_path.shape[1] // 4, mask_path.shape[0] // 4)  # (width, height)
        # resized_dk = cv2.resize(mask_path, new_shape)

        # cv2.imshow("sldfjk", resized_dk)
        # cv2.waitKey()
        

        

    input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask_path, min_area_threshold=min_area_threshold, grids=grids)

    # DEBUG 
    height, width = image.shape[:2]
    input_point_mask = np.zeros((height, width, 3), dtype=np.uint8)
    if save_flag:
        for input_points in input_points_list:
            for input_point in input_points:
                cv2.circle(img=input_point_mask, center=input_point, radius=10, color=(255, 255, 255), thickness=2)
        file_name = f"SAM_input_points.png"
        result, encoded_image = cv2.imencode(".png", input_point_mask)
        encoded_image.tofile(os.path.join(save_path, file_name))

    end_sample_time = time.perf_counter()
    print('sample points time: %f s' % (end_sample_time - start_sample_time))

    tileManager = SamTileManager(
        source_image=image,
        input_points_list=input_points_list,
        input_labels_list=input_labels_list,
    )
    
    tileManager.split_tile()

    image_list, input_points_list_set, input_labels_list_set = tileManager.get_list_of_sam_parameters()


    mask_combine_list = []
    print("start predict each tile")
    count = 0
    predictor = sam_model_setting(config=config)
    
    for local_image, local_points_list, local_labels_list in zip(image_list, input_points_list_set, input_labels_list_set):
        start_predict_time = time.perf_counter()
        show_index = None
        if local_image is None:
            count = count + 1
            end_predict_time = time.perf_counter()
            print(fr"index{count - 1} is none")
            print(f'predict index{count - 1} time: {(end_predict_time - start_predict_time)} s')

        local_mask_path = None
        if save_flag:
            file_name = f"tile_image_{count}.png"
            temp = cv2.cvtColor(local_image, cv2.COLOR_RGB2BGR)
            result, encoded_image = cv2.imencode(".png", temp)
            encoded_image.tofile(os.path.join(save_mask_path, file_name))
            # if count == 13:
            #     show_index = 13

            local_mask_path = os.path.join(save_mask_path, f'{count}')

            if (not os.path.exists(local_mask_path)):
                os.makedirs(local_mask_path, exist_ok=True)
            else:
                for item in os.listdir(local_mask_path):
                    target_file = os.path.join(local_mask_path, item)
                    if os.path.isfile(target_file):
                        os.remove(target_file)

        predictor.set_image(local_image)

        mask_combine_image_after_seg, mask_combine_image, keep_index_list, keep_area_list, keep_bool_list = predict_process(
            image=local_image,
            predictor=predictor,
            input_points_list=local_points_list,
            input_labels_list=local_labels_list,
            usage=usage,
            config=config,
            save_mask_path=local_mask_path,
            index=show_index,
        )
        
        mask_combine_list.append(mask_combine_image)

        if (save_flag):
            file_name = f"combine_{count}_after_seg.png"
            result, encoded_image = cv2.imencode(".png", mask_combine_image_after_seg)
            encoded_image.tofile(os.path.join(save_mask_path, file_name))

            file_name = f"combine_{count}.png"
            result, encoded_image = cv2.imencode(".png", mask_combine_image)
            encoded_image.tofile(os.path.join(save_mask_path, file_name))
            # cv2.imwrite(os.path.join(save_mask_path, file_name), mask_combine_image)
            sam_use.save_keep_index_list(save_mask_path, keep_index_list, keep_area_list, keep_bool_list, f"keep_index_list_{count}.txt")
        
        count = count + 1
        end_predict_time = time.perf_counter()
        print(f'predict index{count - 1} time: {(end_predict_time - start_predict_time)} s')
    
    result = tileManager.combine_result_from_list(mask_combine_list)
    mask_combine_image = Image.fromarray(result)  # Convert to 8-bit grayscale
    
    if save_flag:
        mask_combine_image.save(os.path.join(save_path, 'tile_combine.png'))  # Save as PNG with a unique name
    
    return np.array(mask_combine_image, dtype=np.uint8)

# TODO: run one large road tile data
def predict_large_tile(image_path, output_path, min_area_threshold=0, usage='default', color_usage='all', config=None):
    # create save folder
    save_flag = False
    pixel_cm = config.get_pixel_cm()
    save_path = os.path.join(output_path, usage)
    if (not os.path.exists(save_path)):
        os.makedirs(os.path.normpath(save_path), exist_ok=True)
        
    sample_points_interval = config.get_sample_points_interval(usage)
    
    start_coordinate = (0, 0)
    coordinate_system = 'TWD97'
    pixel_scaling = (0.01, 0.01)

    # check extension is tiff
    if image_path.lower().endswith(('.tif', '.tiff')):
        tiffLoader = TiffLoader(image_path)
        start_coordinate = tiffLoader.get_start_coordinate()
        coordinate_system = tiffLoader.get_coordinate_system()
        pixel_scale = tiffLoader.get_pixel_scale()

        print("tiff info:")
        print(f"start coordinate: {start_coordinate}")
        print(f"coordinate system: {coordinate_system}")
        print(f"pixel scale: {pixel_scale}")

    # split large tile to small tile set
    tile_manager = BigTiffManager(image_path, output_path, tile_size=1024)

    # get each tile starting coordinates
    x_coords, y_coords, x_covers, y_covers = tile_manager.get_split_tile_coords()

    # pass to segment anything and get the result as mask pixel coordinates
    predictor = sam_model_setting(config=config)
    base_tile_path = os.path.join(save_path, 'base_tile')
    if (not os.path.exists(base_tile_path)):
        os.mkdir(base_tile_path)

    coordinate_set = large_tile_seg(tile_manager=tile_manager, 
                                    x_coords=x_coords, y_coords=y_coords,
                                    predictor=predictor, usage=usage, config=config,
                                    output_path=base_tile_path, pixel_cm=pixel_cm,
                                    min_area_threshold=min_area_threshold,
                                    color_usage=color_usage,
                                    sample_points_interval=sample_points_interval,
                                    save_flag=False)
    
    overlap_path = os.path.join(save_path, 'overlap_tile')

    if (not os.path.exists(overlap_path)):
        os.mkdir(overlap_path)

    coordinate_covers_set = large_tile_seg(tile_manager=tile_manager, 
                                x_coords=x_covers, y_coords=y_covers,
                                predictor=predictor, usage=usage, config=config,
                                output_path=overlap_path, pixel_cm=pixel_cm,
                                min_area_threshold=min_area_threshold,
                                color_usage=color_usage,
                                sample_points_interval=sample_points_interval,
                                save_flag=False)
    
    # all mask pixel coordinates
    coordinate_set.update(coordinate_covers_set)

    connected_components = common.connected_components_from_coordinates(coordinate_set)

    json_data = MapJsonData(components=connected_components, start_coordinate=start_coordinate, original_shape=tile_manager.get_shape()[:2])
    json_data.save_to_file(os.path.join(save_path, "map_data.json"))

    height, width, channels = tile_manager.get_shape()
    mask = common.connected_components_to_scaled_mask(connected_components, [height, width], scaled=0.1)

    # imwrtie mask
    cv2.imwrite(os.path.join(save_path, "tile_result.png"), mask)


def large_tile_seg(tile_manager, x_coords, y_coords, predictor, usage, config, output_path, pixel_cm, min_area_threshold, color_usage, sample_points_interval, save_flag=False):
    coordinate_set = set()
    for i, (x_start, y_start) in enumerate(zip(x_coords, y_coords)):
        # get image tile
        image = tile_manager.get_image_tile(y_start, x_start)

        # if image all pixel is zero, skip
        if (image is None or np.sum(image) == 0):
            print(f"image {i} is empty, skip")
            continue

        # filter color mask
        color_filter = ColorFilter(image, output_path, pixel_cm=pixel_cm, config=config, usage=color_usage)
        mask = color_filter.filter_color_mask(color_usage)

        # sample points from mask
        h, w, _ = image.shape
        points_grid = sam_use.build_point_grid_from_real_size(int(pixel_cm), int(image.shape[1]), int(image.shape[0]), int(sample_points_interval))
        input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask, min_area_threshold=min_area_threshold, grids=points_grid)

        # predict sam result
        predictor.set_image(image)
        mask_combine_image_after_seg, mask_combine_image, keep_index_list, keep_area_list, keep_bool_list = predict_process(
            image=image,
            predictor=predictor,
            input_points_list=input_points_list,
            input_labels_list=input_labels_list,
            usage=usage,
            config=config,
            save_mask_path=None,
        )

        ys, xs = np.nonzero(mask_combine_image)
        coordinate_set.update((x + x_start, y + y_start) for x, y in zip(xs, ys))

        if (save_flag):
            file_name = f"combine_{i}.png"
            cv2.imwrite(os.path.join(output_path, file_name), mask_combine_image)
            file_name = f"original_{i}.png"
            cv2.imwrite(os.path.join(output_path, file_name), image)
    return coordinate_set

def predict_each_separate_area(image_path, mask_path, output_path, max_area_threshold=10000, min_area_threshold=0, usage='default', config=None):

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

    pixel_cm = config.get_pixel_cm()
    sample_points_interval = config.get_sample_points_interval(usage)
    grids = sam_use.build_point_grid_from_real_size(int(pixel_cm), int(image.shape[1]), int(image.shape[0]), int(sample_points_interval))
    input_points_list, input_labels_list = sam_use.sample_grid_from_mask(mask_path, min_area_threshold=min_area_threshold, grids=grids, sample_outside=False)

    predictor = sam_model_setting(config=config)
    predictor.set_image(image)
    mask_combine_after_seg, mask_combine, keep_index_list, keep_area_list, keep_bool_list = predict_process(
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
    sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
    if config is not None:
        sam_checkpoint = config.get('Predictor')['checkpoint']
    model_type = "vit_h"

    print("SAM checkpoint:", sam_checkpoint)

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

def run_folder():
    folder = r"D:\GameLab\accident\datas\Xinsheng\0_05m\part2_ortho"

    args = set_args()
    config = None
    config = PredictorConfig(config_path=args.config)

    usage = args.usage
    usage_list = ['default', 'square', 'yellow', 'arrow', 'line']
    color_dict = { 'default': 'default', 'yellow': 'yellow', 'arrow': 'white', 'line': 'default' }
    if config is not None:
        usage_list = config.get(field='Predictor')['usage_list']
        if usage not in usage_list:
            print(f"Usage {usage} not in {usage_list}, use default")
            usage = 'default'
    
    if usage not in usage_list:
        print(f"Usage {usage} not in {usage_list}, use default")
        usage = 'default'
    
    color_usage = args.usage
    if color_usage in color_dict:
        color_usage = color_dict[color_usage]
    print(f"usage {args.usage} map to {color_usage}")

    folder_list = os.listdir(folder)
    for i in range(len(folder_list)):
        index = i + 1
        image_folder = os.path.join(folder, f"ortho_{index}")
        image_file = os.path.join(image_folder, f"ortho_{index}.tif")

        mask_flag = False
        filter = ColorFilter(image_file, image_folder, config=args.config, save_flag=True)

        if args.exist_mask:
            binary_mask = filter.load_existing_mask_binary(color_usage)
            mask_flag = binary_mask is not None
        
        if not mask_flag:
            binary_mask = filter.filter_color_mask(color_usage)
            mask_flag = True
        
        print("start predict image...")

        #predict_each_separate_area(args.image, binary_mask, args.output, usage=usage, config_path=args.config)
        predict_each_sepatate_area_from_image_tile(image_file, binary_mask, image_folder, usage=usage, config=config)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', '-c', type=str, help='Path to the config file', default=None)
    parser.add_argument('--exist_mask', action='store_true', help='Use exist mask to predict')
    parser.add_argument('--usage', type=str, help='Usage of the script', default='default')
    parser.add_argument('--negative-mask', help='a mask indicating areas in result that should be empty')
    parser.add_argument('--bigtiff', action='store_true', help='Use big tiff manager to predict')
    parser.add_argument('--save_log')

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
    config = None
    config = PredictorConfig(config_path=args.config)
    predict_each_separate_area(args.image, args.mask, args.output, config=config)

def main():
    
    args = set_args()
    config = None
    config = PredictorConfig(config_path=args.config)

    usage = args.usage
    usage_list = ['default', 'square', 'yellow', 'arrow', 'line']
    color_dict = { 'default': 'default', 'yellow': 'yellow', 'arrow': 'white', 'line': 'default', 'square': 'default' }
    if config is not None:
        usage_list = config.get(field='Common')['usage_list']
        if usage not in usage_list:
            print(f"Usage {usage} not in {usage_list}, use default")
            usage = 'default'
    else:
        if usage not in usage_list:
            print(f"Usage {usage} not in {usage_list}, use default")
            usage = 'default'
    
    color_usage = args.usage
    if color_usage in color_dict:
        color_usage = color_dict[color_usage]
    print(f"usage {args.usage} map to {color_usage}")

    mask_flag = False
    
    print("start predict image...")

    if (not args.bigtiff):
        filter = ColorFilter(args.image, args.output, config=args.config, save_flag=True, usage=usage)
        print("filter color usage:",  )
        if args.exist_mask:
            binary_mask = filter.load_existing_mask_binary(color_usage)
            mask_flag = binary_mask is not None
        
        if not mask_flag:
            binary_mask = filter.filter_color_mask(color_usage)
            # TEMP
            mask_flag = True
        
        # temp = cv2.resize(binary_mask, (np.array(binary_mask.shape) / 4).astype(np.int32))
        # cv2.imshow("dkfsjld", temp)
        # cv2.waitKey()

        predict_each_sepatate_area_from_image_tile(args.image, binary_mask, args.output, negative_mask=args.negative_mask, usage=usage, config=config)
        #predict_each_separate_area(args.image, binary_mask, args.output, usage=usage, config_path=args.config)
    else:
        predict_large_tile(args.image, args.output, min_area_threshold=0, usage=usage, color_usage=color_usage, config=config)
    
    # color_mask_image = filter.get_color_mask_path()
    # predict_each_sepatate_area_from_image_tile(color_mask_image, binary_mask, args.output, usage=usage, config=config)

if __name__ == "__main__":
    #local_test()
    # run_folder()
    main()