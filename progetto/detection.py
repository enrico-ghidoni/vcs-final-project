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
            gk_size = (3, 3)

            filtered = cv2.GaussianBlur(gray, gk_size, 0)
            otsu_th, th_img = cv2.threshold(filtered, 0, 127, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            edges = cv2.Canny(gray, .8 * otsu_th, otsu_th)

            contours_img = np.copy(rgb_frame)
            contours, hierarchy = cv2.findContours(th_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            print(f'found {len(contours)} contours')

            # estraggo i bounding boxes dai contours
            bboxes_img = np.copy(rgb_frame)
            bboxes = []
            for contour in contours:
                contour_points = np.asarray(contour)
                contour_points = np.squeeze(contour_points)

                print(contour_points)

                ulc_x, ulc_y = np.amin(contour_points, axis=0)
                brc_x, brc_y = np.amax(contour_points, axis=0)
                width = brc_x - ulc_x
                height = brc_y - ulc_y

                box = (ulc_x, ulc_y, width, height)
                bboxes.append(box)

                cv2.circle(bboxes_img, (ulc_x, ulc_y), 5, (0, 255, 0), 2)

            print(bboxes)

            cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 3)

            # disegno i bounding boxes
            for box in bboxes:
                x, y, w, h = box
                cv2.rectangle(bboxes_img, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

            fig, ax = plt.subplots(2, 2, figsize=(24, 14))

            fig.canvas.set_window_title(f'Frame {frame_count}')

            ax[0, 0].imshow(rgb_frame)
            ax[0, 0].set_title('original frame')
            ax[0, 1].imshow(th_img, cmap='gray')
            ax[0, 1].set_title('used to find contours')
            ax[1, 0].imshow(bboxes_img, cmap='gray')
            ax[1, 0].set_title('bounding boxes')
            ax[1, 1].imshow(contours_img)
            ax[1, 1].set_title('contours')
            plt.gcf().canvas.mpl_connect('key_press_event', close)
            plt.show()

        frame_count += 1
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()

    main(**args.__dict__)
