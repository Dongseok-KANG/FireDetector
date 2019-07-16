import cv2
import numpy as np

import sys
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException

api_key = "#YOUR API KEY#"
api_secret = "#YOUR API SECRET#"

params = dict()
params['type'] = 'sms'  # Message type ( sms, lms, mms, ata )
params['to'] = '#Recipients Number#'
params['from'] = '#Sender number#'
params['text'] = '#Message#'

cool = Message(api_key, api_secret)

videofile1 = '#FILE DIRECTORY#'
video = cv2.VideoCapture(videofile1)

i = 0
while True:
    (grabbed, frame) = video.read()

    if not grabbed:
        break

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [0, 50, 50]
    upper = [13, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]

        if area > 400:
            roi = frame[y:y + width, x:x + width]
            dst = cv2.resize(roi, dsize=(28, 28), interpolation=cv2.INTER_AREA)
            dst_1 = dst.reshape(1, 2352)
            predict = sess.run(tf.argmax(logits, 1), feed_dict={X: dst_1, keep_prob: 1})[0]
            if predict == 1:
                cv2.rectangle(frame, (x, y), (x + width, y + width), (0, 0, 255))
                while i == 0:
                    response = cool.send(params)
                    i += 1

    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)
    cv2.imshow('frame', frame)
    cv2.imshow("output", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()