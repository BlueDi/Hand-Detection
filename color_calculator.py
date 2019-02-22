#!/usr/bin/env python

import numpy as np
import cv2


def captureCamera(left=False):
    """
    Creates a color bound based on a ROI
    It analyses the blue square and calculates the maximum, minimum and average HSV values inside the square.
    Those maximum and minimum values will be used to determine the maximum sensibility possible, and the average will be the color bound used to detect the hand.
    Parameters
    ----------
    left : bool, optional
      Set the ROI on the left side of the screen
    """
    cap = cv2.VideoCapture(0)

    outerRectangleXIni = 300
    outerRectangleYIni = 50
    outerRectangleXFin = 550
    outerRectangleYFin = 300
    innerRectangleXIni = 400
    innerRectangleYIni = 150
    innerRectangleXFin = 450
    innerRectangleYFin = 200

    if left:
        move_to_left = 250
        outerRectangleXIni = outerRectangleXIni - move_to_left
        outerRectangleXFin = outerRectangleXFin - move_to_left
        innerRectangleXIni = innerRectangleXIni - move_to_left
        innerRectangleXFin = innerRectangleXFin - move_to_left

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (outerRectangleXIni, outerRectangleYIni),
                      (outerRectangleXFin, outerRectangleYFin), (0, 255, 0), 0)
        cv2.rectangle(frame, (innerRectangleXIni, innerRectangleYIni),
                      (innerRectangleXFin, innerRectangleYFin), (255, 0, 0), 0)
        cv2.putText(frame, 'Please center your hand in the square', (0, 35),
                    font, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            cap.release()
            return None
        elif key != -1:
            roi = frame[innerRectangleYIni +
                        1:innerRectangleYFin, innerRectangleXIni +
                        1:innerRectangleXFin]
            display_result(roi)
            approved = wait_approval()
            if approved:
                break
            cv2.destroyAllWindows()

    cap.release()
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
    upper = np.array(
        [hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
    h = hsvRoi[:, :, 0]
    s = hsvRoi[:, :, 1]
    v = hsvRoi[:, :, 2]
    hAverage = np.average(h)
    sAverage = np.average(s)
    vAverage = np.average(v)

    hMaxSensibility = max(abs(lower[0] - hAverage), abs(upper[0] - hAverage))
    sMaxSensibility = max(abs(lower[1] - sAverage), abs(upper[1] - sAverage))
    vMaxSensibility = max(abs(lower[2] - vAverage), abs(upper[2] - vAverage))

    cv2.destroyAllWindows()
    return np.array([[hAverage, sAverage, vAverage],
                     [hMaxSensibility, sMaxSensibility, vMaxSensibility]])


def display_result(roi):
    """Draws images of the selected ROI"""
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_result = np.concatenate((roi, hsvRoi))
    cv2.imshow('ROI Result', roi_result)


def wait_approval():
    """Checks if User wants the selected ROI"""
    approval = False
    key = cv2.waitKey(0)
    if key != -1 and key != ord('n'):
        approval = True
    return approval


def main():
    """Main function of the app"""
    captureCamera()


if __name__ == '__main__':
    main()

