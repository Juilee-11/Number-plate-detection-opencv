import json


def Groundtooth_data():
    path_json = "/home/edlabadkar/Downloads/test_coco(2).json"
    with open(path_json) as f:
        data = json.load(f)

    img_data = data['images']

    img_annotation = data['annotations']

    plate_data = dict()
    gt_data = list()

    for one_img in img_data:
        img_id = one_img['id']
        file_name = one_img['file_name']
        box_coordinate = list()
        for one_annot in img_annotation:
            ann_img_id = one_annot['image_id']
            # print(ann_img_id, img_id)
            if ann_img_id == img_id:
                box_coordinate = one_annot['bbox']
                plate_data[file_name]=box_coordinate
        # gt_data.append(box_coordinate)
    # write the json#
    return plate_data
# plate_dict = Groundtooth_data()
# print("plate dict -->",plate_dict)

def show_img(img):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom offset dimensions
    cv2.resizeWindow("output", 400, 300)
    cv2.imshow("output", img)
    cv2.waitKey(0)
    return


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # print(interArea)
    # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou*100


def get_iou_numberplate(all_gt, all_number_plate):
    # gt --> all 51 gt images = [[x1,y1, w,h]...]
    # all_number plate = [[[x1,y1,x2,y2],[]],.....[]]
    # det_iou_list = [iou_val]

    max_iou_list = list()

    for one_gt_plate, det_nplate in zip(all_gt, all_number_plate):

        gt_plate = [one_gt_plate[0], one_gt_plate[1], one_gt_plate[0] + one_gt_plate[2],
                    one_gt_plate[1] + one_gt_plate[3]]

        det_iou_list = list()

        for one_plate in det_nplate:
            iou_val = bb_intersection_over_union(gt_plate, one_plate)
            det_iou_list.append(iou_val)
        try:
            one_max_iou = max(det_iou_list)
            max_iou_list.append(one_max_iou)
        except ValueError:
            max_iou_list.append('NONE')

    return max_iou_list
