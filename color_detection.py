#!/usr/bin/env python

import numpy as np
import cv2


def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    """Thresholds the image to the color bounds"""
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', hsv_image)
    mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
    return mask


def getContours(binary_image):
    """Calculates the contours of the image"""
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_hand_contour(binary_image, rgb_image, contours):
    """Draws the contours to the image"""
    black_image = np.zeros([binary_image.shape[0], binary_image.shape[1], 3],
                           'uint8')
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if (area > 130):
            cv2.drawContours(rgb_image, [c], -1, (150, 250, 150), 1)
            cv2.drawContours(black_image, [c], -1, (150, 250, 150), 1)
            cx, cy = get_contour_center(c)
            cv2.circle(rgb_image, (cx, cy), (int)(radius), (0, 0, 255), 1)
            cv2.circle(black_image, (cx, cy), (int)(radius), (0, 0, 255), 1)
            cv2.circle(black_image, (cx, cy), 5, (150, 150, 255), -1)
            print('Area: {}, Perimeter: {}'.format(area, perimeter))
    print('Number of contours: {}'.format(len(contours)))
    cv2.imshow('RGB Image Contours', rgb_image)
    cv2.imshow('Black Image Contours', black_image)


def get_contour_center(contour):
    """Calcualtes the center of the contour"""
    M = cv2.moments(contour)
    cx = -1
    cy = -1
    if (M['m00'] != 0):
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy


def draw_contours(lower_bound_color, upper_bound_color):
    """Calculates the contours and draws them in the picture"""
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        binary_image_mask = filter_color(frame, lower_bound_color,
                                         upper_bound_color)
        contours = getContours(binary_image_mask)
        draw_hand_contour(binary_image_mask, frame, contours)

        key = cv2.waitKey(10)
        if key == ord('q'):
            video_capture.release()
            break

    cv2.destroyAllWindows()


def main():
    """Main function of the app"""
    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    draw_contours(lower_color, upper_color)


if __name__ == '__main__':
    main()

