import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--video',
                    help='Absolute path for video file to process.')
parser.add_argument('--output-file',
                    help='Absolute path for file containing regions detected as paintings.')
parser.add_argument('--skip',
                    help='Number of frames to skip after every processed frame.',
                    type=int,
                    default=0)
parser.add_argument('--debug',
                    help='Enable debug logging',
                    action='store_true')


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


def main(video, output_file, skip, debug=False):
    print(f'main starting with video {video}')
    cap = cv2.VideoCapture(video)
    print(cap.isOpened())
    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu_th, th_img = cv2.threshold(filtered, 0, 127, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        edges = cv2.Canny(gray, .8 * otsu_th, otsu_th)

        fig, ax = plt.subplots(2, 2, figsize=(24, 14))

        ax[0, 0].imshow(gray, cmap='gray')
        ax[0, 0].set_title('grayscale image')
        ax[0, 1].imshow(th_img, cmap='gray')
        ax[0, 1].set_title('after otsu thresholding')
        ax[1, 0].imshow(edges, cmap='gray')
        ax[1, 0].set_title('edges with canny')

        # plt.subplot(221)
        # plt.imshow(gray, cmap='gray', aspect='auto')
        # plt.subplot(222)
        # plt.imshow(th_img, cmap='gray', aspect='auto')
        # plt.subplot(223)
        # plt.imshow(edges, cmap='gray', aspect='auto')
        plt.gcf().canvas.mpl_connect('key_press_event', close)
        plt.show()

        # key = cv2.waitKey(0)
        # while key not in [ord('q'), ord('k')]:
        #     key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()

    main(**args.__dict__)
