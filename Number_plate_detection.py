import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract as tess
from PIL import Image

from coco_json_GT import Groundtooth_data, get_iou_numberplate


def preprocess(img, dir_path):
    imgBlurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(f'{dir_path}img_blurred.jpg', imgBlurred)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    cv2.imwrite(f'{dir_path}sobelx.jpg', sobelx)
    ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f'{dir_path}threshold_img.jpg', threshold_img)
    return threshold_img


def cleanPlate(plate, dir_path):
    print("CLEANING PLATE. . .")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh = cv2.dilate(gray, kernel, iterations=1)
    _, thresh = cv2.threshold(thresh, 120, 255, cv2.THRESH_BINARY)

    # mask = cv2.bitwise_and(gray, thresh)

    __, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    countor_img = cv2.drawContours(plate, contours, -1, (0, 255, 0), 3)
    cv2.imwrite(f'{dir_path}plate_countor_img{time.time()}.jpg', countor_img)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        # print(max_index)
        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        # print(max_cntArea)
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate, None

        cleaned_final = thresh[y:y + h, x:x + w]
        det_coordinate = [x, y, w, h]

        return cleaned_final, det_coordinate

    else:
        return plate, None


def extract_contours(threshold_img, ori_image, dir_path):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()

    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    cv2.imwrite(f'{dir_path}morphed_img.jpg', morph_img_threshold)

    __, contours, hierarchy = cv2.findContours(morph_img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 30 and h < 50:
            cv2.rectangle(ori_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imwrite(f'{dir_path}img_countor_img.jpg', ori_img)

    return contours


def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    # print(ratio)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 15 * aspect * 10  # minimum area
    max = 125 * aspect * 125  # maximum area
    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True


def isMaxWhite(plate):
    avg = np.mean(plate)
    if (avg >= 115):
        return True
    else:
        return False


def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect
    if (width > height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle
    if angle > 10:
        return False
    if height == 0 or width == 0:
        return False

    area = height * width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True


def cleanAndRead(img, contours, dir_path):
    count = 0
    rect_plate_list = list()
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        if validateRotationAndRatio(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]
            if (isMaxWhite(plate_img)):

                clean_plate, rect = cleanPlate(plate_img, dir_path)
                rect_plate_list.append([x, y, x + w, y + h])
                if rect:
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x + x1, y + y1, w1, h1
                    cv2.imwrite(f'{dir_path}pplate_img_{count}.jpg', clean_plate)
                    count += 1
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(f'{dir_path}final_img.jpg', img)

    return rect_plate_list, count


if __name__ == '__main__':
    print(f'\n Start Process !!!')
    imgpath_all = "/home/edlabadkar/all_frame0/"
    all_images = glob.glob(imgpath_all + '/*.jpg')

    all_frames = list()
    all_number_plate = list()
    all_plate_count = list()
    gt_plate_box = list()
    count = 0
    groundtooth_data = Groundtooth_data()
    for one_img in all_images:
        frame_name = (one_img.split('/')[-1]).split('.')[0]
        gt_box = groundtooth_data[one_img.split('/')[-1]]

        dir_path = f'/home/edlabadkar/output6/{frame_name}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        img = cv2.imread(one_img)
        cv2.imwrite(f'{dir_path}ori_img.jpg', img)
        ori_img = img.copy()
        threshold_img = preprocess(img, dir_path)
        contours = extract_contours(threshold_img, ori_img, dir_path)
        count += 1
        number_plate, plate_count = cleanAndRead(img, contours, dir_path)

        all_frames.append(frame_name)
        all_number_plate.append(number_plate)
        all_plate_count.append(plate_count)
        gt_plate_box.append(gt_box)

    max_iou_values = get_iou_numberplate(gt_plate_box, all_number_plate)

    data = {
        'Frame_name': all_frames,
        'GT_Plate_coordinate': gt_plate_box,
        'Detected_plate_coordinate': all_number_plate,
        'Plate_count_detected': all_plate_count,
        'IOU value': max_iou_values,
    }

    df = pd.DataFrame(data, columns=['Frame_name',
                                     'GT_Plate_coordinate',
                                     'Detected_plate_coordinate',
                                     'Plate_count_detected',
                                     'IOU value'])

    df.to_csv(r'/home/edlabadkar/result.csv', index=False)
    print(f'\n End Process !!!')
