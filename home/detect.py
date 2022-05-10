import time
import cv2
import argparse
import numpy as np
from imutils.video import VideoStream
import imutils


# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     return output_layers

# # Ham ve cac hinh chu nhat va ten class


# def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
#     label = str(classes[class_id])
#     cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), 2)
#     cv2.putText(img, label, (x - 10, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)


LIST_LABELS = {
    # 'person': 'Đây là người bạn nhé',
    'bicycle': 'Đây là cái xe đạp',
    'car': 'Đây là ô tô nha',
    'motorbike': 'Đây là xe máy nhé',
    'aeroplane': 'Đây là máy bay nhé',
    'bus': 'Đây là xe buýt nhé',
    'train': 'Đây là tàu hỏa',
    'truck': 'Đây là xe tải',
    'boat': 'Đây là thuyền',
    'traffic light': 'Đây là đèn giao thông',
    'fire hydrant': 'Đây là trụ nước chữa cháy',
    'stop sign': 'Đây là cảnh báo dừng lại',
    'parking meter': 'nơi đỗ xe',
    'bench': 'đây là ghế dài',
    'bird': 'đây là chim',
    'cat': 'đây là mèo',
    'dog': 'đây là chó',
    'horse': 'đây là ngựa',
    'sheep': 'đây là cừu',
    'cow': 'đây là bò',
    'elephant': 'đây là voi',
    'bear': 'đây là gấu',
    'zebra': 'đây là ngựa vằn',
    'giraffe': 'đây là hươu cao cổ',
    'backpack': 'đây là cặp sách',
    'umbrella': 'đây là cái ô',
    'handbag': 'đây là túi xách',
    'tie': 'đây là cà vạt',
    'suitcase': 'đây là va li',
    'frisbee': 'đây là đĩa ném',
    'skis': 'đây là đĩa ván trượt',
    'snowboard': 'đây là ván trượt tuyết',
    'sports ball': 'đây là bóng đá',
    'kite': 'đây là cái diều',
    'baseball bat': 'đây là mũ bóng chày',
    'baseball glove': 'đây là găng tay chơi bóng chày',
    'skateboard': 'đây là ván trượt',
    'surfboard': 'đây là ván lướt sóng',
    'tennis racket': 'đây là vợt tennis',
    'bottle': 'đây là cái bình',
    'wine glass': 'đây là ly rượu',
    'cup': 'đây là cái cốc',
    'fork': 'đây là cái dĩa',
    'knife': 'đây là con dao',
    'spoon': 'đây là cái thìa',
    'bowl': 'đây là cái bát',
    'banana': 'đây là qủa chuối',
    'apple': 'đây là quả táo',
    'sandwich': 'đây là bánh mỳ kẹp',
    'orange': 'đây là quả cam',
    'broccoli': 'đây là bông súp lơ',
    'carrot': 'đây là cà rốt',
    'hot dog': 'đây là xúc xích',
    'pizza': 'đây là bánh pi da',
    'donut': 'đây là bánh vòng ',
    'cake': 'đây là cái bánh',
    'chair': 'đây là cái ghế',
    'sofa': 'đây là ghé sô pha',
    'pottedplant': 'đây là cây',
    'bed': 'đây là cái giường',
    'diningtable': 'đây là bàn ăn',
    'toilet': 'đây là toi lét',
    'tvmonitor': 'đây là Ti vi',
    'laptop': 'đây là máy tính xách tay',
    'mouse': 'đây là con chuột',
    'remote': 'đây là cái điều khiển',
    'keyboard': 'đây là bàn phím',
    'cell phone': 'đây là điện thoại',
    'microwave': 'đây là lò vi sóng',
    'oven': 'đây là lò nướng',
    'toaster': 'đây là lò nướng bánh mỳ',
    'sink': 'đây là bồn rửa chén',
    'refrigerator': 'đây là tủ lạnh',
    'book': 'đây là quyển sách',
    'clock': 'đây là đồng hồ',
    'vase': 'đây là bình hoa',
    'scissors': 'đây là cái kéo',
    'teddy bear': 'đây là gấu tét đi',
    'hair drier': 'đây là mấy sấy tóc',
    'toothbrush': 'đây là bàn chải đánh răng',
}
last_object_name = ''
# Ham tra ve output layer
CONFIG_PATH = "/home/chessie/Documents/ITNUT/ITNUT/home/object_detect/yolov3-coco/yolov3.cfg"
WEIGHTS_PATH = "/home/chessie/Documents/ITNUT/ITNUT/home/object_detect/yolov3-coco/yolov3.weights"
LABELS_PATH = "/home/chessie/Documents/ITNUT/ITNUT/home/object_detect/yolov3-coco/coco.names"


def render_class():
    classes = None
    with open(LABELS_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    classes = render_class()
    label = str(classes[class_id])
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), 2)

    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)


def detect(image_path):
    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None
    with open(LABELS_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    end = time.time()
    print("YOLO Execution time: " + str(end-start))
    return(class_ids)
