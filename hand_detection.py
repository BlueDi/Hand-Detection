#!/usr/bin/env python

import numpy as np
import cv2
import argparse
import color_calculator as cc
import color_detection as cd
import video_detection as vd


def analyse_args():
    """Parses the args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', default=None, help='Place ROI on the left')
    parser.add_argument(
        '-l', '--left', action='store_true', help='Place ROI on the left')
    parser.add_argument(
        '-s',
        '--shot',
        action='store_true',
        help='Not video, just a single shot')
    return parser.parse_args()


def main():
    """Main function of the app"""
    args = analyse_args()
    video_capture = cv2.VideoCapture(0)
    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    while True:
        _, frame = video_capture.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, 'Welcome', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('VCOM Project', frame)
        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            video_capture.release()
            break

    if key == ord('v'):
        try:
            avg_color, max_sensibility = cc.captureCamera(args.left)
            vd.start(
                avg_color,
                max_sensibility,
                video=not args.shot,
                path=args.input,
                left=args.left)
        except TypeError:
            print 'Did not calculate the color bound.'
    elif key == ord('h'):
        cd.draw_contours(lower_color, upper_color)


if __name__ == '__main__':
    main()

