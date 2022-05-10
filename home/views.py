import imutils
from imutils.video import VideoStream
import argparse
import time
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpRequest
from django.views.generic import TemplateView
from django.views.generic import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
import base64
from matplotlib.font_manager import json_dump
import numpy as np
from PIL import Image
import io
from django.views.decorators.csrf import csrf_exempt
import json
import cv2


# Create your views here.
MEDIA_PATH = '/home/chessie/Documents/ITNUT/ITNUT_1/home/media/'


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
    0: 'Đây là người bạn nhé',
    1: 'Đây là cái xe đạp',
    2: 'Đây là ô tô nha',
    3: 'Đây là xe máy nhé',
    4: 'Đây là máy bay nhé',
    5: 'Đây là xe buýt nhé',
    6: 'Đây là tàu hỏa',
    7: 'Đây là xe tải',
    8: 'Đây là thuyền',
    9: 'Đây là đèn giao thông',
    10: 'Đây là trụ nước chữa cháy',
    11: 'Đây là cảnh báo dừng lại',
    12: 'nơi đỗ xe',
    13: 'đây là ghế dài',
    14: 'đây là chim',
    15: 'đây là mèo',
    16: 'đây là chó',
    17: 'đây là ngựa',
    18: 'đây là cừu',
    19: 'đây là bò',
    20: 'đây là voi',
    21: 'đây là gấu',
    22: 'đây là ngựa vằn',
    23: 'đây là hươu cao cổ',
    24: 'đây là cặp sách',
    25: 'đây là cái ô',
    26: 'đây là túi xách',
    27: 'đây là cà vạt',
    28: 'đây là va li',
    29: 'đây là đĩa ném',
    30: 'đây là đĩa ván trượt',
    31: 'đây là ván trượt tuyết',
    32: 'đây là bóng đá',
    33: 'đây là cái diều',
    34: 'đây là mũ bóng chày',
    35: 'đây là găng tay chơi bóng chày',
    36: 'đây là ván trượt',
    37: 'đây là ván lướt sóng',
    38: 'đây là vợt tennis',
    39: 'đây là cái bình',
    40: 'đây là ly rượu',
    41: 'đây là cái cốc',
    42: 'đây là cái dĩa',
    43: 'đây là con dao',
    44: 'đây là cái thìa',
    45: 'đây là cái bát',
    46: 'đây là qủa chuối',
    47: 'đây là quả táo',
    48: 'đây là bánh mỳ kẹp',
    49: 'đây là quả cam',
    50: 'đây là bông súp lơ',
    51: 'đây là cà rốt',
    52: 'đây là xúc xích',
    53: 'đây là bánh pi da',
    54: 'đây là bánh vòng ',
    55: 'đây là cái bánh',
    56: 'đây là cái ghế',
    57: 'đây là ghé sô pha',
    58: 'đây là cây',
    59: 'đây là cái giường',
    60: 'đây là bàn ăn',
    61: 'đây là toi lét',
    62: 'đây là Ti vi',
    63: 'đây là máy tính xách tay',
    64: 'đây là con chuột',
    65: 'đây là cái điều khiển',
    66: 'đây là bàn phím',
    67: 'đây là điện thoại',
    68: 'đây là lò vi sóng',
    69: 'đây là lò nướng',
    70: 'đây là lò nướng bánh mỳ',
    71: 'đây là bồn rửa chén',
    72: 'đây là tủ lạnh',
    73: 'đây là quyển sách',
    74: 'đây là đồng hồ',
    75: 'đây là bình hoa',
    76: 'đây là cái kéo',
    77: 'đây là gấu tét đi',
    78: 'đây là mấy sấy tóc',
    79: 'đây là bàn chải đánh răng',
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
    start = time.time()
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


def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def saveimage(base64_string):
    imgdata = (base64.b64decode(base64_string))
    # I assume you have a way of picking unique filenames
    filename = "/home/chessie/Documents/ITNUT/ITNUT_1/home/media/image.jpg"
    with open(filename, 'wb') as f:
        f.write(imgdata)


def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list


class IndexView(View):
    def get(self, request):
        return HttpResponse('hello')


@method_decorator(csrf_exempt, name='dispatch')
class UploadImage(View):
    def post(self, request):
        # get the base64 encoded string
        response_data = []
        data = request.body.decode('utf-8')
        data = json.loads(str(data))
        im_b64 = (data["image"])
        name = (data["name"])
        saveimage(im_b64)
        id_labels_detected = detect(
            "/home/chessie/Documents/ITNUT/ITNUT_1/home/media/image.jpg")
        unique_list = unique(id_labels_detected)
        print(unique_list)
        # data_json = json.dumps(str(unique_list))
        for id in unique_list:
            if id != 0:
                response_data.append(LIST_LABELS[id])
        # print(data_json)
        return JsonResponse(str(unique_list), safe=False, status=200)
