import cv2
import argparse
import os
import numpy as np
import sys
sys.path.append("./utils")
import common

# filter out the color which not close to given color_list
class ColorFilter:
    def __init__(self, input_path, output_path, 
                 color_list=[[225, 225, 225], [112, 206, 244]], 
                 color_threshold=80, area_threshold=50, 
                 config=None, save_flag=False):
        
        # init member variables
        self.input_path = input_path
        self.output_path = output_path
        self.color_list = color_list
        self.color_threshold = color_threshold
        self.area_threshold = area_threshold
        self.save_flag = save_flag
        self.color_label = []
        if config is not None:
            self._set_config_param(config)

        self.image = None
        self.mask_color_image = None
        self.mask_binary_image = None
        self.label_image_list = None

        # check input_path is file
        if not os.path.isfile(input_path):
            raise ValueError(f"Input path {input_path} is not a valid file.")
        
        self.image = cv2.imread(input_path)  # Load in grayscale
        self.image = common.enhance_brightness_saturation(self.image)
        cv2.imwrite(os.path.join(self.output_path, 'enhanced_image.jpg'), self.image)

        if self.image is None:
            raise ValueError(f"Could not read image from {input_path}. Check the file path and format.")

    def filter_color_mask(self, color_str='all'):
        # Apply color filter
        self.mask_color_image, self.label_image_list = self._keep_pixel_color(self.image, threshold=self.color_threshold)

        # Save the modified image
        if self.save_flag:
            filename = self._get_color_mask_path()
            cv2.imwrite(filename, self.mask_color_image)

        gray_image = cv2.cvtColor(self.mask_color_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.mask_binary_image = self._clean_small_area_from_mask(binary_image, self.area_threshold)
        if (self.label_image_list is not None):
            for i in range(len(self.label_image_list)):
                self.label_image_list[i] = self._clean_small_area_from_mask(self.label_image_list[i], self.area_threshold)

        if self.save_flag:
            filename = self._get_binary_mask_path()
            cv2.imwrite(filename, self.mask_binary_image)
            if (self.label_image_list is not None) and (self.color_label is not None):
                for i in range(len(self.label_image_list)):
                    filename = self._get_binary_color_mask_path(self.color_label[i])
                    cv2.imwrite(filename, self.label_image_list[i])
        
        index = -1
        # index = where self.color_label == color_str, if no match label index = -1
        if color_str != 'all':
            index = self.color_label.index(color_str) if color_str in self.color_label else -1
        # if index == -1, return the mask binary image

        if index == -1:
            # return the mask binary image
            return self.mask_binary_image.copy()
        else:
            # return the label image
            return self.label_image_list[index].copy()

    def load_existing_mask_binary(self, color_str='all'):
        if (color_str not in self.color_label):
            mask_path = self._get_binary_mask_path()
        else:
            mask_path = self._get_binary_color_mask_path(color_str)

        # check if mask_path exists
        if not os.path.exists(mask_path) or not os.path.isfile(mask_path):
            print(f"Mask path {mask_path} does not exist or is not a valid file.")
            return None

        # load mask image
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"Could not read mask image from {mask_path}. Check the file path and format.")
            return None
        _, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    def get_label_image_from_color(self, color_str='white'):
        # check if color_str is in color_label list
        if color_str not in self.color_label:
            return None

        # get the index of the color_str in color_label list
        index = self.color_label.index(color_str)

        # return the label image
        return self.label_image_list[index]

    def _get_color_mask_path(self):
        filename = os.path.join(self.output_path, 'filter_color.jpg')
        return filename

    def _get_binary_mask_path(self):
        filename = os.path.join(self.output_path, 'filter_color_binary.jpg')
        return filename

    def _get_binary_color_mask_path(self, color_str='_'):
        filename = os.path.join(self.output_path, f'filter_color_binary_{str(color_str)}.jpg')
        return filename

    def _keep_pixel_color(self, image, threshold=80):
        # grab the image dimensions
        h, w = image.shape[:2]

        reshaped_image = image.reshape(-1, 3)  # (H*W, 3)
        colors = np.array(self.color_list)     # (N, 3)

        # check each pixel distance
        dists = np.linalg.norm(reshaped_image[:, None, :] - colors[None, :, :], axis=2)

        # find closest color index
        closest_indices = np.argmin(dists, axis=1)
        closest_distances = np.min(dists, axis=1)

        within_threshold = closest_distances <= threshold
        
        target = np.zeros_like(reshaped_image)
        label_image_list = [np.zeros((h * w,), dtype=np.uint8) for _ in range(len(self.color_list))]

        # default image
        target[within_threshold] = reshaped_image[within_threshold]

        # color image
        for i in range(len(self.color_list)):
            mask = within_threshold & (closest_indices == i)
            label_image_list[i][mask] = 255

        # reshape to original image
        target = target.reshape(h, w, 3)
        label_image_list = [label.reshape(h, w) for label in label_image_list]

        # return the thresholded image
        return target, label_image_list
    
    def _clean_small_area_from_mask(self, mask, threshold=50):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep_labels = np.zeros(num_labels, dtype=np.uint8)  # label 0 is background

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= threshold:
                keep_labels[i] = 1
                
        filtered_mask = keep_labels[labels]

        filtered_mask = (filtered_mask * 255).astype(np.uint8)
        return filtered_mask
    
    def _set_config_param(self, config):
        if config is not None:
            # load config file
            config = common.load_config(config_path=config, field='ColorFilter')
            self.color_list = config['color_list']
            self.color_threshold = config['color_threshold']
            self.area_threshold = config['area_threshold']
            self.save_flag = config['save_flag']
            self.color_label = config['color_label']
            print("color filter use config:", config)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    parser.add_argument('--config', '-c', type=str, help='Path to the config file', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    filter = ColorFilter(args.image, args.output, config=args.config, save_flag=True)
    filter.filter_color_mask()
    