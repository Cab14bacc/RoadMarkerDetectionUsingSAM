# External Libraries
import cv2
import numpy as np
# Internal Libraries
from . import color_filter
from . import predictor
from . import spline_test
# Internal Config/JsonData Classes
from .utils.configUtils.predictorConfig import PredictorConfig
from .utils.configUtils.splineTestConfig import SplineTestConfig

def road_line_detection(image_path, config_path, negative_mask_path):
    """
    params:
        image_path: path to image 
        config_path: path to config
        negative_mask_path: path to mask indicating areas to ignore
    return 
        json_data: class SplineJsonData, contains the detected road lines  
    """
    # region Predictor and ColorFilter ===============================================


    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    predictor_config = PredictorConfig(config_path=config_path)
    if predictor_config.get_all_config() is None:
        raise RuntimeError(
            f"Failed to load configuration from '{config_path}'. "
            "The configuration object is empty or None.")


    usage = "line"
    usage_list = ['default', 'square', 'yellow', 'arrow', 'line']
    color_dict = { 'default': 'default', 'yellow': 'yellow', 'arrow': 'white', 'line': 'default', 'square': 'default' }

    common_config_content = predictor_config.get(field='Common')
    
    if not common_config_content:
        raise ValueError("Missing required configuration section: 'Common'")
    
    usage_list = common_config_content.get('usage_list')
    if usage_list is None:
        raise ValueError("Missing required configuration field: 'usage_list'")
    
    if usage not in usage_list:
        raise ValueError(f"Invalid usage '{usage}'. Allowed: {usage_list}")
    
    color_usage = "line"
    if color_usage in color_dict:
        color_usage = color_dict[color_usage]
    else:
        raise ValueError("Missing required color usage field: 'line'")

    print(f"usage {usage} map to {color_usage}")

    save_flag = False
    filter = color_filter.ColorFilter(image, 
                                      None, 
                                      config=predictor_config, 
                                      save_flag=save_flag, 
                                      usage=usage)

 
    binary_mask = filter.filter_color_mask(color_usage)    
    tile_combine_image = predictor.predict_each_sepatate_area_from_image_tile(image, binary_mask, None, negative_mask_path, usage=usage, config=predictor_config)
    # endregion Predictor and ColorFilter ===============================================
    # region SplineTest =================================================================
    spline_test_config = SplineTestConfig(config_path=config_path)
    
    if spline_test_config.get_all_config() is None:
        raise RuntimeError(
            f"Failed to load configuration from '{config_path}'. "
            "The configuration object is empty or None.")
    
    json_data = spline_test.keep_main_spline(tile_combine_image, image, None, spline_test_config)
    # endregion SplineTest =================================================================

    return json_data


if __name__ == "__main__":
    json_data = road_line_detection(r"./input/新生南_cropped_0_01/cropped_0001.tif", 
                        r"./configs/config_0_01_no_save.yml", 
                        r"./output/新生南_cropped_0_01/cropped_0001_tif/MarkerClassifier/result.png")

