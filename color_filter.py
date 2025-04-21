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
        
        if config is not None:
            self._set_config_param(config)

        self.image = None
        self.mask_color_image = None
        self.mask_binary_image = None

        # check input_path is file
        if not os.path.isfile(input_path):
            raise ValueError(f"Input path {input_path} is not a valid file.")
        
        self.image = cv2.imread(input_path)  # Load in grayscale
        if self.image is None:
            raise ValueError(f"Could not read image from {input_path}. Check the file path and format.")

    def filter_color_mask(self):
        # Apply color filter
        self.mask_color_image = self._keep_pixel_color(self.image, threshold=self.color_threshold)

        # Save the modified image
        if self.save_flag:
            filename = os.path.join(self.output_path, 'filter_color.jpg')
            cv2.imwrite(filename, self.mask_color_image)

        gray_image = cv2.cvtColor(self.mask_color_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.mask_binary_image = self._clean_small_area_from_mask(binary_image, self.area_threshold)

        if self.save_flag:
            filename = os.path.join(self.output_path, 'filter_color_binary.jpg')
            cv2.imwrite(filename, self.mask_binary_image)
        
        # return the copy of mask binary image
        return self.mask_binary_image.copy()

    def _keep_pixel_color(self, image, threshold=80):
        # grab the image dimensions
        h, w = image.shape[:2]

        # copy image
        target = image.copy()

        # loop over the image pixel by pixel, keep the color in color_list
        # distances: the difference between the color and the color in color_list
        for y in range(0, h):
            for x in range(0, w):
                color = image[y, x]
                target[y, x], distances = self._find_close_color(image[y, x], np.array(self.color_list), threshold=threshold)
        # return the thresholded image
        return target
    
    # set black if the color is not in color_list
    def _find_close_color(self, pixel, colors, threshold=None):
        distances = np.sqrt(np.sum((colors-pixel)**2,axis=1))
        index_of_smallest = np.where(distances==np.amin(distances))

        # check distance small than threshold. If not return black
        if threshold is not None and distances[index_of_smallest[0]][0] > threshold:
            return np.array([0, 0, 0]), distances
        return pixel, distances
    
    def _clean_small_area_from_mask(self, mask, threshold=50):
        '''
        # contour and connectedComponent seems to be the same result
        # keep contour method for maybe future use on shape

        # Apply connected components
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # for i in range(1, num_labels):
        #     x, y, w, h, area = stats[i]
        #     if area < 100:  # filter out small blobs
        #         mask[labels == i] = 0
        # return mask
        '''
        # Remove objects that are too small or have odd aspect ratios that don’t match typical road markers:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < threshold:
                cv2.drawContours(mask, [cnt], -1, 0, -1)

        return mask
    
    def _set_config_param(self, config):
        if config is not None:
            # load config file
            config = common.load_config(config_path=config, field='ColorFilter')
            self.color_list = config['color_list']
            self.color_threshold = config['color_threshold']
            self.area_threshold = config['area_threshold']
            self.save_flag = config['save_flag']
            print("color filter use config:", config)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    filter = ColorFilter(args.image, args.output, save_flag=True)
    filter.filter_color_mask()
    