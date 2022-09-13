import cv2
import pickle
import numpy as np
from tracker import *

import tensorflow as tf

# print(tf.test.is_gpu_available())
with tf.device('/GPU:1'):

    tracker = EuclideanDistTracker()

    accuracy = 0.5
    vehicle = [0, 2, 5, 7]
    Car = (0, 0, 255)
    People = (0, 255, 0)
    bike = (255, 0, 0)
    Max_dis = 3.34
    count = 0
    FPS = 1
    Font = cv2.FONT_HERSHEY_PLAIN  # font type
    rec1 = np.array([[500, 410], [1098, 503], [1024, 713], [300, 553]], np.int32)
    history = {}

    def in_polylines(target_x, target_y):
        if (cv2.pointPolygonTest(rec1, (target_x, target_y), False)) != -1:
            return True
        return False


    Net = cv2.dnn.readNet(r'F:\Danhai\YOLO\yolov4.weights', r'F:\Danhai\YOLO\yolov4.cfg')  # module

    classes = []
    with open(r'F:\Danhai\YOLO\coco.names', 'r') as F:  # 省去close
        classes = F.read().splitlines()  # \n分隔

    CAP = cv2.VideoCapture(r'F:\Danhai\monitor\A3\20210125\A3-1_2021-01-25-07_00_2021-01-25-08_00.mp4')

    history = {}
    while CAP.isOpened():
        ret, IMG = CAP.read()
        if type(IMG) == type(None):
            break
        # cv2.line(IMG, (500, 410), (1098, 503), (0, 0, 0), 1)
        # cv2.line(IMG, (1098, 503), (1024, 713), (0, 0, 0), 1)
        # cv2.line(IMG, (1024, 713), (300, 553), (0, 0, 0), 1)
        # cv2.line(IMG, (300, 553), (500, 410), (0, 0, 0), 1)

        check = {}
        cv2.polylines(IMG, pts=[rec1], isClosed=True, color=cv2.COLOR_BGR2GRAY, thickness=2)

        # IMG = IMG[300:700, 100:1500]
        if (count % FPS) == 0:

            Height, Width, ret = IMG.shape

            blob = cv2.dnn.blobFromImage(IMG, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
            Net.setInput(blob)
            output_layers_names = Net.getUnconnectedOutLayersNames()
            layerOutputs = Net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []
            detections = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]  # 第 6 ~ 最後
                    class_id = np.argmax(scores)
                    if class_id == 2 or class_id == 5 or class_id == 7:
                        confidence = scores[class_id]
                        if confidence > accuracy:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            W = int(detection[2] * Width)
                            H = int(detection[3] * Height)
                            x = int(center_x - W / 2)
                            y = int(center_y - H / 2)
                            if in_polylines(center_x, center_y):
                                boxes.append([x, y, W, H])
                                confidences.append((float(confidence)))
                                class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, accuracy, 0.4)
            boxes = tracker.update(boxes)
            # print(len(boxes))
            # print(indexes.flatten())
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, W, H, ID = boxes[i]
                    label = str(classes[class_ids[i]])
                    # confidence = str(round(confidences[i], 2))
                    center_x = int(x + W / 2)
                    center_y = int(y + H / 2)
                    cv2.circle(IMG, (center_x, center_y), 8, Car, -1)
                    cv2.rectangle(IMG, (x, y), (x + W, y + H), Car, 2)
                    cv2.putText(IMG, label + " " + str(ID), (x, y + 20), Font, 2, (255, 255, 255), 2)
                    if ID in history:
                        check[ID] = history[ID] + 1
                    else:
                        check[ID] = 1

            IMG = cv2.resize(IMG, (1080, 720))
            cv2.imshow("Video", IMG)
            key = cv2.waitKey(1)
            if key == 27:
                break
            history.update(check)
            print(history)
        count += 1

    CAP.release()
    cv2.destroyAllWindows()
    file = open("A3.pkl", "wb")
    pickle.dump(history, file)
    file.close()
