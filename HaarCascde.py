import numpy as np
import cv2
import argparse
import imutils
from myqueue import myqueue
from frames import frames
import time
#to do speed up FPS

def main():
    fullbody_cascade = cv2.CascadeClassifier(
        '/Users/Aaron/Documents/College/Fourth_Year/Final_Year_Project/Datasets/body10/haarcascade_fullbody.xml')

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

    fps = frames().start()

    while camera.more():
        frame = camera.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(frame, "Queue Size: {}".format(camera.Q.qsize()),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        fullbody = fullbody_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in fullbody:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        cv2.imshow("HaarCascade", frame)
        cv2.waitKey(1)
        fps.update()
        k = cv2.waitKey(27) & 0xff
        if k == 27:
            break
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    camera.stop()


if __name__ == '__main__':
    main()