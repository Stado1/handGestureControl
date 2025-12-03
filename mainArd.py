import sys
import cv2
import torch
import numpy as np

# Add local YOLOv5 path
sys.path.append('yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

import pyfirmata2
import time

import controlFunctions


## load the arduino bord
port = pyfirmata2.Arduino.AUTODETECT  # auto-detects /dev/cu.* or COM*
board = pyfirmata2.Arduino(port)
controlFunctions.setup_pins(board)


# Load model
device = select_device('cpu')
model = DetectMultiBackend(
    'handGestureRecognizer.pt',
    device=device,
    dnn=False,
)

stride = model.stride
imgsz = check_img_size(640, s=stride)



# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam")
    exit()

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame

    # preprocess
    img_resized = cv2.resize(img, (imgsz, imgsz))
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB and HWC -> CHW
    img_resized = np.ascontiguousarray(img_resized)

    im = torch.from_numpy(img_resized).to(device)
    im = im.float() / 255.0
    im = im.unsqueeze(0)  # add batch dimension (1,3,H,W)

    # Inference
    pred = model(im)
    pred = non_max_suppression(pred, 0.25, 0.45)

 
    threshold = 0.5

    # Draw only the bounding box with the highest confidence
    for det in pred:
        if len(det):

            # Sort detections by confidence (descending)
            det = det[det[:, 4].argsort(descending=True)]

            # Take only the top-1 result
            det = det[:1]
            
#            print(det)
#            print(det[0][4].item())
            
            if (det[0][4].item() > threshold):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    label = f"{model.names[int(cls)]}: {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                
                                
#                print(det[0][-1].item())
#               0=B, 1=F, 2=L, 3=R, 4=S
                direction = int(det[0][-1].item())
                
                if (direction == 4):
                    controlFunctions.stop(board)
                elif (direction == 0):
                    controlFunctions.backward(board)
                elif (direction == 1):
                    controlFunctions.forward(board)
                elif (direction == 2):
                    controlFunctions.left(board)
                elif (direction == 3):
                    controlFunctions.right(board)
                else:
                    controlFunctions.stop(board)
                


    # Display frame
    cv2.imshow("YOLOv5", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
