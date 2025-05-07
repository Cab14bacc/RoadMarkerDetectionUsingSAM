import os
import numpy as np
import cv2

import common

class SAMMaskSelector:
    def __init__(self, config):
        
        self.config = config
        self.ready = False
        if self.config is not None:
            self.ready = True
        self.selected_mask = None

    def use_default_config(self):
        self.config = { 
            'Predictor': {
                'max_area_threshold': [250000, 750000],
                'min_area_threshold': [0, 0],
                'color_threshold': 120,
                'pixel_cm': 5
            } 
        }
        self.ready = True

    def selector(self, mask, index, usage='default'):
        if usage == 'default' or usage == 'yellow':
            return self.default_selector(mask, index, usage)
        elif usage == 'line':
            return self.line_selector(mask, index)
        return False
    
    def default_selector(self, mask, index, usage='default'):
        config = self.config['Predictor']
        pixel_cm = config['pixel_cm']
        if usage == 'yellow':
            min_area_threshold = config['min_area_threshold'][1] 
            max_area_threshold = config['max_area_threshold'][1]
        else:
            min_area_threshold = config['min_area_threshold'][0]
            max_area_threshold = config['max_area_threshold'][0]

        min_area_threshold = min_area_threshold / pixel_cm / pixel_cm
        max_area_threshold = max_area_threshold / pixel_cm / pixel_cm

        area_size = np.sum(mask)
        if ((area_size < min_area_threshold) or (area_size > max_area_threshold)):
            # change from bool to int
            mask_int = mask.astype(np.uint8) * 255
            # check if is long line
            if (common.analyze_line_mask(mask_int, pixel_cm=pixel_cm, index=index)):
                return True
            else:
                print(f"mask{index} area {area_size} larger or smaller than threshold, skip")
                return False
        
        return True

    def line_selector(self, mask, index, ratio=10):
        config = self.config['Predictor']
        pixel_cm = config['pixel_cm']
        min_area_threshold = config['min_area_threshold'][1] / pixel_cm / pixel_cm
        max_area_threshold = config['max_area_threshold'][1] / pixel_cm / pixel_cm
        pixel_cm = config['pixel_cm']

        area_size = np.sum(mask)
        if (area_size > max_area_threshold):
            return False
        mask_int = mask.astype(np.uint8) * 255

        keep_flag, result_mask = common.analyze_all_line_mask(mask_int, ratio=8, pixel_cm=pixel_cm)
        self.selected_mask = result_mask
        if (keep_flag):
            return True
        # if (common.analyze_line_mask(mask_int, ratio=2)):
        #     return True
        else:
            #print(f"mask{index} is not a line, skip")
            return False
    
    def get_selected_mask(self):
        return self.selected_mask