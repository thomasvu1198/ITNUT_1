import cv2
import requests
from imutils.video import VideoStream
import imutils
import time
import base64
import json
# from yolo_utils import speak


def frame_to_base64(frame):
    return base64.b64encode(frame)


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

api = 'http://192.168.0.108:8000/upload/'
MEDIA_PATH = '/home/chessie/Documents/ITNUT/ITNUT/home/media'

vidcap = cv2.VideoCapture(0)
success, image = vidcap.read()
image = imutils.resize(image, width=320)
count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    success, image = cv2.imencode('.jpg', image)
    print('Read a new frame: ', success)
    image_file = "frame%d.jpg" % count
    im_bytes = base64.b64encode(image).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_bytes, "name": image_file})
    response = requests.post(api, data=payload, headers=headers)
    try:
        data = (response.content)
        data = (str(data).replace("b'", "").replace("'",  ""). replace(
            '"', ''). replace("[", "").replace("]", ""). replace(" ", ""))
        print(list(data.split(",")))
        for id in list(data.split(",")):
            if int(id) != 0 :
                print(LIST_LABELS[int(id)])
                speak(LIST_LABELS[int(id)])
    except requests.exceptions.RequestException:
        print('err: ', response.text)
    count += 1
    # time.sleep(4)
# requests.post(url, files=files)
