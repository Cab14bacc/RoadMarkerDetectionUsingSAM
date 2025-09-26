import cv2
import argparse
import os
import numpy as np
import sys
sys.path.append("./utils")
import utils.common as common

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
        
        self.image = cv2.imread(input_path)
        if self.image is None:
            raise ValueError(f"Could not read image from {input_path}. Check the file path and format.")

    def filter_color_mask(self):
        # Apply color filter
        self.mask_color_image = self._keep_pixel_color(self.image, threshold=self.color_threshold)

        # Save the modified image
        if self.save_flag:
            filename = self._get_color_mask_path()
            cv2.imwrite(filename, self.mask_color_image)

        gray_image = cv2.cvtColor(self.mask_color_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.mask_binary_image = self._clean_small_area_from_mask(binary_image, self.area_threshold)

        if self.save_flag:
            filename = self._get_binary_mask_path()
            cv2.imwrite(filename, self.mask_binary_image)
        
        # return the copy of mask binary image
        return self.mask_binary_image.copy()

    #def load_existing_mask_binary(self, mask_path):

    def load_existing_mask_binary(self):

        mask_path = self._get_binary_mask_path()
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

    def _get_color_mask_path(self):
        filename = os.path.join(self.output_path, 'filter_color.jpg')
        return filename

    def _get_binary_mask_path(self):
        filename = os.path.join(self.output_path, 'filter_color_binary.jpg')
        return filename

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
        index_of_smallest = np.nonzero(distances==np.amin(distances))[0]
        
        # check distance small than threshold. If not return black
        if threshold is not None and distances[index_of_smallest][0] > threshold:
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





def match_arrow_template(img, template, template_scale, max_angle=360, step=2):
    best_match = None
    best_angle = 0
    best_score = -1  # start with a low score
    best_template_img = None

    template_h, template_w = template.shape[:2]
    template_longest_side = np.hypot(template_h, template_w).astype(np.int32)

    new_img_size = int(template_longest_side * template_scale)

    image_h, image_w = img.shape[:2]

    new_img = img
    new_img_width = new_img_size if image_w < new_img_size else image_w
    new_img_height = new_img_size if image_h < new_img_size else image_h

    need_resize = image_h < new_img_size or image_w < new_img_size

    if need_resize:
        new_img = np.zeros((new_img_height, new_img_width), np.uint8)
        center_x, center_y = new_img_width // 2, new_img_height // 2 
        
        top_left = center_x - image_w // 2, center_y - image_h // 2
        bot_right = top_left[0] + image_w, top_left[1] + image_h
        
        new_img[top_left[1]:bot_right[1], top_left[0]:bot_right[0]] = img


    # Loop through angles from 0 to max_angle, with the specified step
    for angle in range(0, max_angle, step):
        # Rotate the template by the current angle
        rotated_template = rotate_image(template, angle, template_scale)
        # cv2.imshow("rotated_template", rotated_template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Perform template matching
        result = cv2.matchTemplate(new_img, rotated_template, cv2.TM_CCORR_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        # print(f"current angle: {angle}, current score: {score}")

        # top_left = max_loc
        # h, w = rotated_template.shape[:2]
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # new_img_copy = new_img.copy()
        # new_img_copy = cv2.cvtColor(new_img_copy, cv2.COLOR_GRAY2RGBA)
        # cv2.rectangle(new_img_copy, top_left, bottom_right, (255, 255, 0), 2)

        # cv2.imshow("", rotated_template)
        # # Show the result
        # cv2.imshow("Matched Image", new_img_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Get the best match location and the maximum correlation value
        
        # Update the best match if the current score is better
        if score > best_score:
            best_score = score
            best_match = max_loc
            best_angle = angle
            best_template_img = rotated_template

    print(f"Best match found at location: {best_match}")
    print(f"Best rotation angle: {best_angle} degrees")
    print(f"Best matching score: {best_score:.4f}")
    # cv2.imshow("best template img", best_template_img)
    # cv2.imshow("Matched Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Return the best match location and the angle
    return best_match, best_angle, best_score, best_template_img

def rotate_image(image, angle, scale):
    """Rotates an image by a specified angle."""
    # Get the image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Calculate new image dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(((h * sin) + (w * cos)))
    new_h = int(((h * cos) + (w * sin)))

    # Adjust rotation matrix to consider translation
    M[0, 2] += (new_w / 2) - (center[0])
    M[1, 2] += (new_h / 2) - (center[1])

    # Get the rotation matrix
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Path to the image file', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path to save the output image', default='./')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = set_args()
    filter = ColorFilter(args.image, args.output, save_flag=True, config="./config.yml")
    filter.filter_color_mask()


    height, width = filter.mask_binary_image.shape[:2]
    cnts, _ = cv2.findContours(filter.mask_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = np.zeros((len(cnts), height, width), dtype=np.uint8)
    template = cv2.imread(r"C:\Users\Leo\Documents\GitRepos\OSMAPI\Assets\ArrowTemplates\FLArrow.png", 0)
    
    for i, cnt in enumerate(cnts):
        masks[i] = cv2.drawContours(masks[i], [cnt], contourIdx=-1, color=255, thickness=-1)

    for i, mask in enumerate(masks):
        print(f"matching mask {i}")
        best_match, best_angle, best_score, best_template_img = match_arrow_template(mask, template, template_scale=0.2)
        
        best_template_h, best_template_w = best_template_img.shape[:2]

        longest_best_template_side = best_template_h if best_template_h > best_template_w else best_template_w
        
        need_resize = longest_best_template_side > height or longest_best_template_side > width
        new_mask_h, new_mask_w =  height, width

        new_mask = mask
        if need_resize:
            new_mask_h = longest_best_template_side if longest_best_template_side > height else height
            new_mask_w = longest_best_template_side if longest_best_template_side > width else width

            new_mask = np.zeros((new_mask_h, new_mask_w), np.uint8)
            new_center = (new_mask_w // 2,  new_mask_h // 2)
            new_top = new_center[1] - height // 2
            new_left = new_center[0] - width // 2
            new_bot = new_top + height
            new_right = new_left + width
            
            new_mask[new_top:new_bot, new_left:new_right] = mask

        
        new_template_img = np.zeros((new_mask_h, new_mask_w), np.uint8)
        new_center = (new_mask_w // 2,  new_mask_h // 2)
        new_top = new_center[1] - best_template_h // 2
        new_left = new_center[0] - best_template_w // 2
        new_bot = new_top + best_template_h
        new_right = new_left + best_template_w
        
        new_template_img[new_top:new_bot, new_left:new_right] = best_template_img

        output = np.concatenate([new_template_img, new_mask], axis=1)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        cv2.putText(output, f"{best_score}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


        cv2.imwrite(os.path.join("output", "vis_" + str(i) + ".png"), output)

    
    