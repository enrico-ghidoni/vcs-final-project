import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random as rng

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
            gk_size = (gf_size, gf_size)

            filtered = cv2.GaussianBlur(gray, gk_size, 0)
            otsu_th, th_img = cv2.threshold(filtered, 0, 127, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            edges = cv2.Canny(filtered, 0.5 * otsu_th, otsu_th)

            padded_img = np.pad(th_img, (5, 5), mode='constant', constant_values=(1, 1))

            contours_img = np.copy(rgb_frame)
            contours, hierarchy = cv2.findContours(padded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            print(f'found {len(contours)} contours')

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])

            bboxes_img = np.copy(rgb_frame)
            for i in range(len(contours)):
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                cv2.drawContours(contours_img, contours_poly, i, color, thickness=3)
                cv2.rectangle(bboxes_img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                             (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 3)

            # estraggo i bounding boxes dai contours
            # bboxes_img = np.copy(rgb_frame)
            # bboxes = []
            # for contour in contours:
            #     contour_points = np.asarray(contour)
            #     contour_points = np.squeeze(contour_points)
            #
            #     print(contour_points)
            #
            #     ulc_x, ulc_y = np.amin(contour_points, axis=0)
            #     brc_x, brc_y = np.amax(contour_points, axis=0)
            #     width = brc_x - ulc_x
            #     height = brc_y - ulc_y
            #
            #     box = (ulc_x, ulc_y, width, height)
            #     bboxes.append(box)
            #
            #     cv2.circle(bboxes_img, (ulc_x, ulc_y), 5, (0, 255, 0), 2)
            #
            # print(bboxes)

            # disegno i bounding boxes
            # for box in bboxes:
            #     x, y, w, h = box
            #     cv2.rectangle(bboxes_img, (x, y), (x + w, y + h),
            #                   (0, 255, 0), 2)

            if debug:
                fig, ax = plt.subplots(2, 3, figsize=(24, 14))

                fig.canvas.set_window_title(f'Frame {frame_count}')

                ax[0, 0].imshow(rgb_frame)
                ax[0, 0].set_title('original frame')
                ax[0, 1].imshow(filtered, cmap='gray')
                ax[0, 1].set_title('blurred image')
                ax[0, 2].imshow(edges, cmap='gray')
                ax[0, 2].set_title('canny edge detection')
                ax[1, 0].imshow(th_img, cmap='gray')
                ax[1, 0].set_title('otsu thresholding')
                ax[1, 1].imshow(contours_img)
                ax[1, 1].set_title('contours')
                ax[1, 2].imshow(bboxes_img)
                ax[1, 2].set_title('bounding boxes')
                plt.gcf().canvas.mpl_connect('key_press_event', close)
                plt.show()
            else:
                out = cv2.cvtColor(bboxes_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'Output', out)
                cv2.waitKey(75)

        frame_count += 1
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()

    main(**args.__dict__)
