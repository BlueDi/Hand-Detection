#!/usr/bin/env python

import numpy as np
import cv2
import math
import traceback


def hand_detection(lower_bound_color, upper_bound_color, left=False):
    video_capture = cv2.VideoCapture(0)

    while True:
        try:
            _, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            kernel = np.ones((3, 3), np.uint8)

            if left:
                roi = frame[100:300, 100:300]
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
            else:
                roi = frame[50:300, 300:550]
                cv2.rectangle(frame, (300, 50), (550, 300), (0, 255, 0), 0)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
            mask = cv2.dilate(mask, kernel, iterations=5)
            erosion = cv2.erode(mask, kernel, iterations=5)
            erosion = cv2.GaussianBlur(erosion, (5, 5), 100)

            _, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)

            cnt = max(contours, key=lambda x: cv2.contourArea(x))

            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            hull = cv2.convexHull(cnt)

            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)

            arearatio = ((areahull - areacnt) / areacnt) * 100

            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            l = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                d = (2 * ar) / a

                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)
                cv2.line(roi, start, end, [0, 255, 0], 2)

            l += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l == 1:
                if areacnt < 2000:
                    cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2,
                                (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    if arearatio < 12:
                        cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255),
                                    3, cv2.LINE_AA)
                    elif arearatio < 17.5:
                        cv2.putText(frame, 'Fixe', (0, 50), font, 2,
                                    (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255),
                                    3, cv2.LINE_AA)
            elif l == 2:
                cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            elif l == 3:
                if arearatio < 27:
                    cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3,
                                cv2.LINE_AA)
            elif l == 4:
                cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            elif l == 5:
                cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            elif l == 6:
                cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255),
                            3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'reposition', (10, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)

            show_results(mask, erosion, frame)

        except Exception as e:
            print e
            pass

        key = cv2.waitKey(10)
        if key == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break


def show_results(mask, erosion, frame):
    combine_masks = np.concatenate((mask, erosion), axis=0)
    height, _, _ = frame.shape
    _, width = combine_masks.shape
    masks_result = cv2.resize(combine_masks, dsize=(width, height))
    masks_result = cv2.cvtColor(masks_result, cv2.COLOR_GRAY2BGR)
    result_image = np.concatenate((frame, masks_result), axis=1)
    cv2.imshow('Hand Detection', result_image)


def main():
    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    hand_detection(lower_color, upper_color)


if __name__ == '__main__':
    main()
