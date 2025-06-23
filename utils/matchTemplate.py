
import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression
from utils.common import get_image

def matchTemplateImpl(draw_image, src_image, template_image, threshold=0.5):
    result = cv2.matchTemplate(src_image, template_image, cv2.TM_CCOEFF_NORMED)
    # result = cv2.matchTemplate(src_image, template_image, cv2.TM_CCORR_NORMED)
    
    loc = np.where(result >= threshold)

    rh, rw = template_image.shape
    boxes = []
    scores = []
    flag = []
    for pt in zip(*loc[::-1]):
        box = [pt[0], pt[1], pt[0] + rw, pt[1] + rh]
        boxes.append(box)
        scores.append(result[pt[1], pt[0]])
    return boxes, scores

def rotate_image_bound(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to shift the image center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=0)
    return rotated

def filter_boxes(boxes, scores, angles, threshold=0.1):
    # Filter boxes and angles based on the threshold using non_max_suppression
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    angles_np = np.array(angles)
    # print shape of three
    #print(f"boxes shape: {boxes_np.shape}, scores shape: {scores_np.shape}, angles shape: {angles_np.shape}")
    picks = non_max_suppression(boxes_np, probs=scores_np, overlapThresh=threshold)

    angles_result = []
    for pick in picks:
        i = np.where((boxes_np == pick).all(axis=1))[0][0]
        angles_result.append(angles_np[i])

    return picks, angles_result

def match_template_all_rotation(image, template_path, output_path, scale_height=50, threshold=0.6, angle_step=5, pre_name='', save_flag=False):
    # Load the source image
    img_rgb = get_image(image)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"

    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_rgb  # Already grayscale

    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    # scale template to height 50 pixel as same ratio
    scale = scale_height / template.shape[0]
    template = cv2.resize(template, (int(template.shape[1] * scale), scale_height))

    w, h = template.shape[::-1]
    match_count = 0

    boxes = []
    scores = []
    angles = []
    for angle in range(0, 360, angle_step):
        # Rotate the template
        rotated = rotate_image_bound(template, angle)
        # extend to boxes and scores list
        new_boxes, new_scores = matchTemplateImpl(img_gray, img_gray, rotated, threshold)
        boxes.extend(new_boxes)
        scores.extend(new_scores)
        # append the angle to angles list with the same length as boxes and scores
        for _ in range(len(new_boxes)):
            angles.append(angle)
    

    result_boxes, result_angles = filter_boxes(boxes, scores, angles, 0.3)

    for (x1, y1, x2, y2) in result_boxes:
        match_count += 1
        # cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    result_image = np.zeros_like(img_rgb)

    # crop according to the boxes
    cropped_images = []
    # for loop result_boxes and angles
    for (x1, y1, x2, y2), angle in zip(result_boxes, result_angles):

        # Crop the image
        # cropped_image = img_rgb[y1:y2, x1:x2]
        # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        gray_crop = img_gray[y1:y2, x1:x2]
        _, thresh = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Find the largest contour
        cnt = max(contours, key=cv2.contourArea)

        black = np.zeros_like(gray_crop)
        cv2.drawContours(black, [cnt], -1, (255, 255, 255), -1)
        result_image[y1:y2, x1:x2] = black


    # print(f"Total matches found: {match_count}")

    if (save_flag):
        # Save the result as res_{scale_heght}_{threshold}.png
        filename = f'{pre_name}_{scale_height}_{threshold}.png'
        filename = os.path.join(output_path, filename)
        cv2.imwrite(filename, result_image)
    return result_image