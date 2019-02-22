#!/usr/bin/env python

import numpy as np
import cv2
import math
import traceback


def nothing(x):
    pass


def apply_sensibility(avg_color, newHSens, newSSens, newVSens, maxSensibility):
    """
    Applies sensibility values for each value of HSV, taking into account the maximum sensibility possible.
    It analyses the parameters and executes the hand detection accordingly.
    Parameters
    ----------
    avg_color : array
      The average of HSV values to be detected
    newHSens : int
      Percentage of sensibility to apply to Hue
    newSSens : int
      Percentage of sensibility to apply to Saturation
    newVSens : int
      Percentage of sensibility to apply to Value
    maxSensibility : array
      The maximum error margin of HSV values to be detected
    """
    hSens = (newHSens * maxSensibility[0]) / 100
    SSens = (newSSens * maxSensibility[1]) / 100
    VSens = (newVSens * maxSensibility[2]) / 100
    lower_bound_color = np.array(
        [avg_color[0] - hSens, avg_color[1] - SSens, avg_color[2] - VSens])
    upper_bound_color = np.array(
        [avg_color[0] + hSens, avg_color[1] + SSens, avg_color[2] + VSens])
    return np.array([lower_bound_color, upper_bound_color])


def start(avg_color, max_sensibility, video=True, path=None, left=False):
    """
    Initializes the detection process.
    It analyses the parameters and executes the hand detection accordingly.
    Parameters
    ----------
    avg_color : array
      The average of HSV values to be detected
    max_sensibility : array
      The maximum error margin of HSV values to be detected
    video : bool, optional
      False if single image
      True if video stream
    path : str, optional
      Path for the image to be analysed
    left : bool, optional
      Set the ROI on the left side of the screen
    """

    # change this value to better adapt to environment light (percentage values)
    hSensibility = 100
    sSensibility = 100
    vSensibility = 100

    apply_sensibility(avg_color, hSensibility, sSensibility, vSensibility,
                      max_sensibility)

    cv2.namedWindow('Hand Detection')
    cv2.createTrackbar('HSensb', 'Hand Detection', hSensibility, 100, nothing)
    cv2.createTrackbar('SSensb', 'Hand Detection', sSensibility, 100, nothing)
    cv2.createTrackbar('VSensb', 'Hand Detection', vSensibility, 100, nothing)

    if path != None:
        frame = cv2.imread(path)
        hand_detection(frame, lower_bound_color, upper_bound_color, left)
        cv2.waitKey(0)
    else:
        video_capture = cv2.VideoCapture(0)

        while True:
            try:
                _, frame = video_capture.read()
                frame = cv2.flip(frame, 1)

                # get values from trackbar
                newHSens = cv2.getTrackbarPos('HSensb', 'Hand Detection')
                newSSens = cv2.getTrackbarPos('SSensb', 'Hand Detection')
                newVSens = cv2.getTrackbarPos('VSensb', 'Hand Detection')

                # and apply the new sensibility values
                lower_bound_color, upper_bound_color = apply_sensibility(
                    avg_color, newHSens, newSSens, newVSens, max_sensibility)

                hand_detection(frame, lower_bound_color, upper_bound_color,
                               left)

            except Exception as e:
                print e
                pass

            if not video:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

            key = cv2.waitKey(10)
            if key == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                break


def hand_detection(frame, lower_bound_color, upper_bound_color, left):
    """
    Initializes the detection process.
    It analyses the parameters and executes the hand detection accordingly.
    Parameters
    ----------
    frame : array-like
      The frame to be analysed
    lower_bound_color : array
      The min of HSV values to be detected
    upper_bound_color : array
      The max of HSV values to be detected
    left : bool, optional
      Set the ROI on the left side of the screen
    """
    kernel = np.ones((3, 3), np.uint8)

    if left:
        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    else:
        roi = frame[50:300, 300:550]
        cv2.rectangle(frame, (300, 50), (550, 300), (0, 255, 0), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    binary_mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
    mask = cv2.dilate(binary_mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 90)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        l = analyse_defects(cnt, roi)
        analyse_contours(frame, cnt, l + 1)
    except ValueError:
        pass

    show_results(binary_mask, mask, frame)


def analyse_defects(cnt, roi):
    """
    Calculates how many convexity defects are on the image.
    A convexity defect is a area that is inside the convexity hull but does not belong to the object.
    Those defects in our case represent the division between fingers.
    Parameters
    ----------
    cnt : array-like
      Contour of max area on the image, in this case, the contour of the hand
    roi : array-like
      Region of interest where should be drawn the found convexity defects
    """
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    l = 0
    if defects is not None:
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
    return l


def analyse_contours(frame, cnt, l):
    """
    Writes to the image the signal of the hand.
    The hand signals can be the numbers from 0 to 5, the 'ok' signal, and the 'all right' symbol.
    The signals is first sorted by the number of convexity defects. Then, if the number of convexity defects is 1, 2, or 3, the area ratio is to be analysed.
    Parameters
    ----------
    frame : array-like
      The frame to be analysed
    cnt : array-like
      Contour of max area on the image, in this case, the contour of the hand
    l : int
      Number of convexity defects
    """
    hull = cv2.convexHull(cnt)

    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    arearatio = ((areahull - areacnt) / areacnt) * 100

    font = cv2.FONT_HERSHEY_SIMPLEX
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2,
                        (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame, 'Fixe', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            else:
                cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
    elif l == 2:
        cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 3:
        if arearatio < 27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3,
                        cv2.LINE_AA)
    elif l == 4:
        cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 5:
        cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 6:
        cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3,
                    cv2.LINE_AA)
    else:
        cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3,
                    cv2.LINE_AA)


def show_results(binary_mask, mask, frame):
    """
    Shows the image with the results on it.
    The image is a result of a combination of the image with the result on it, the original captured ROI, and the ROI after optimizations.
    Parameters
    ----------
    binary_mask : array-like
      ROI as it is captured
    mask : array-like
      ROI after optimizations
    frame : array-like
      Frame to be displayed
    """
    combine_masks = np.concatenate((binary_mask, mask), axis=0)
    height, _, _ = frame.shape
    _, width = combine_masks.shape
    masks_result = cv2.resize(combine_masks, dsize=(width, height))
    masks_result = cv2.cvtColor(masks_result, cv2.COLOR_GRAY2BGR)
    result_image = np.concatenate((frame, masks_result), axis=1)
    cv2.imshow('Hand Detection', result_image)


def main():
    """Main function of the app"""
    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    start(lower_color, upper_color)


if __name__ == '__main__':
    main()

