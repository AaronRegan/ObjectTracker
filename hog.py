# import the necessary packages
from __future__ import print_function
from non_max_suppression import non_max_suppression
from myqueue import myqueue
from frames import frames
from object import Object
import numpy as np
import argparse
import datetime
import imutils
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
# otherwise, we are reading from a video file
else:
    print("[INFO] starting video file thread...")
    camera = myqueue(args["video"]).start()
    time.sleep(1.0)

i = 0
centerX = 0
centerY = 0
objList = []
meas = []
pred = []
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03


def onPed(x, y):
    global mp, meas
    mp = np.array([[np.float32(x)], [np.float32(y)]])
    meas.append((x, y))


def kalPredict(mp):
    global tp, pred
    kalman.correct(mp)
    tp = kalman.predict()
    pred.append((int(tp[0]), int(tp[1])))


def paint(tp, xA, yA, xB, yB):
    global frame, pred
    # cv2.circle(frame, ((tp[0]), (tp[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(frame, ((tp[0]) - ((xB - xA) / 2), (tp[1]) + (yB - yA) / 2),
                  (((tp[0]) + ((xB - xA) / 2)), ((tp[1]) - (yB - yA) / 2)), (0, 0, 255), 2)

fps = frames().start()
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# loop over the image paths
while camera.more():
    frame = camera.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # start = datetime.datetime.now()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(32, 32), scale=1.05)
    # print("[INFO] detection took: {}".format(
    #(datetime.datetime.now() - start).total_seconds()))
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        i = i+1
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        centerX = (xB + xA) / 2
        centerY = (yB + yA) / 2
        obj = Object(centerX, centerY, i)
        objList.append(obj)
        onPed(centerX, centerY)
        kalPredict(mp)
        paint(tp, xA, yA, xB, yB)

    cv2.putText(frame, "Queue Size: {}".format(camera.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    peds_found = "Found " + str(len(pick)) + " Pedestrians"
    cv2.putText(frame, peds_found, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the output images
    cv2.imshow("HOG", frame)
    cv2.waitKey(1)
    fps.update()
    k = cv2.waitKey(27) & 0xff
    if k == 27:
        break
fps.stop()
for objects in range(len(objList) - 1):
    print(str(objList[objects]))
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
camera.stop()
