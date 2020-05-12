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
parser.add_argument('--onein',
                    help='Evaluate one frame every N frames.',
                    type=int,
                    default=1)
parser.add_argument('--debug',
                    help='Enable debug logging',
                    action='store_true')


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


def main(video, output_file, onein, debug=False):
    frame_count = 1

    print(f'main starting with video {video}')
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame_count % onein == 0:

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            std = np.std(gray)
            gf_size = int(np.ceil(3 * std) // 2 * 2 + 1)
            gk_size = (2 * gf_size + 1, 2 * gf_size + 1)

            filtered = cv2.GaussianBlur(gray, gk_size, 0)
            otsu_th, th_img = cv2.threshold(filtered, 0, 127, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            edges = cv2.Canny(gray, .8 * otsu_th, otsu_th)

            contours_img = np.copy(rgb_frame)
            contours, hierarchy = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)

            fig, ax = plt.subplots(2, 2, figsize=(24, 14))

            fig.canvas.set_window_title(f'Frame {frame_count}')

            ax[0, 0].imshow(rgb_frame)
            ax[0, 0].set_title('original frame')
            ax[0, 1].imshow(th_img, cmap='gray')
            ax[0, 1].set_title('after otsu thresholding')
            ax[1, 0].imshow(filtered, cmap='gray')
            ax[1, 0].set_title('after gaussian filter')
            ax[1, 1].imshow(contours_img)
            ax[1, 1].set_title('contours')

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
        frame_count += 1
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()

    main(**args.__dict__)
