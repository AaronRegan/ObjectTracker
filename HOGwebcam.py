# import the necessary packages
from __future__ import print_function
from webcamThread import webcamThread
from frames import frames
from non_max_suppression import non_max_suppression
from munkres import Munkres, print_matrix
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

i = 0
objList = []
meas = []
pred = []
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction


def onPed(x, y):
    global mp, meas
    mp = np.array([[np.float32(x)], [np.float32(y)]])
    meas.append((x, y))


def updateKalman(mp):
    global pred, tp
    kalman.correct(mp)
    tp = kalman.predict()
    pred.append((int(tp[0]), int(tp[1])))

def paint(tp, xA, yA, xB, yB):
    global frame, pred
    cv2.circle(frame, ((tp[0]), (tp[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(frame, ((tp[0]) - ((xB - xA) / 2), (tp[1]) + (yB - yA) / 2),
                  (((tp[0]) + ((xB - xA) / 2)), ((tp[1]) - (yB - yA) / 2)), (0, 0, 255), 2)


# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = webcamThread(src=0).start()
fps = frames().start()

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over some frames...this time using the threaded stream
while (True):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(32, 32), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        i = i + 1
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # print((((xB - xA) / 2), ((yB - yA)) / 2))
        # print(xA)
        # print(yA)
        # print(xB)
        # print(yB) for debugging
        centerX = (xB + xA) / 2
        centerY = (yB + yA) / 2
        # obj = object(centerX,centerY,i)
        # objList.append(obj)
        onPed(centerX, centerY)
        updateKalman(mp)
        paint(tp, xA, yA, xB, yB)

    peds_found = "Found " + str(len(pick)) + " Pedestrians"
    cv2.putText(frame, peds_found, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # show the output images
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        k = cv2.waitKey(27) & 0xff
        if k == 27:
            break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print(objList)
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
