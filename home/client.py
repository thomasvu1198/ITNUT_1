import cv2
import requests
from imutils.video import VideoStream
import imutils
import time
import base64
import json

def frame_to_base64(frame):
    return base64.b64encode(frame)

api = 'http://192.168.0.108:8000/upload/'
MEDIA_PATH = '/home/chessie/Documents/ITNUT/ITNUT/home/media'

vidcap = cv2.VideoCapture(0)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    image_file = "frame%d.jpg" % count
    # with open(image_file, "rb") as f:
    #     im_bytes = f.read()
    im_bytes = frame_to_base64(image)
    print(im_bytes)
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64, "name": image_file})
    response = requests.post(api, data=payload, headers=headers)
    # try:
    #     data = response.content
    #     print(data)
    # except requests.exceptions.RequestException:
    #     print('err: ', response.text)
    count += 1
    time.sleep(4)
# requests.post(url, files=files)
