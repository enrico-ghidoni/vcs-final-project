import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random as rng
import json

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
                    help='Show detailed comparison of image processing, if off only bounding '
                         'boxes applied to original frame will be shown.',
                    action='store_true')
parser.add_argument('--noshow',
                    help=' Do not show results while processing.',
                    action='store_true')


def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)


def main(video, output_file, onein, debug=False, noshow=False):
    frame_count = 0
    bboxes_json = {}

    print(f'main starting with video {video}')
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % onein == 0 and ret:

            bboxes_json[frame_count] = []

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            std = np.std(gray)
            gf_size = int(np.ceil(3 * std) // 2 * 2 + 1)
            gk_size = (gf_size, gf_size)
            gk_size = (13, 13)

            filtered = cv2.GaussianBlur(gray, gk_size, 0)
            otsu_th, th_img = cv2.threshold(filtered, 180, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            # cv2.imshow('thresholded image', th_img)
            # cv2.waitKey()

            dilated_img = cv2.dilate(th_img, None, iterations=15)
            # cv2.imshow('dilated image', dilated_img)
            # cv2.waitKey()

            num_labels, labels_im = cv2.connectedComponentsWithAlgorithm(dilated_img, 8, cv2.CV_32S, cv2.CCL_GRANA)
            print(num_labels)
            print(len(labels_im))
            label_hue = np.uint8(179 * labels_im / np.max(labels_im))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            # cv2.imshow('connected components image', labeled_img)
            # cv2.waitKey()

            # eroded_img = cv2.dilate(dilated_img, None, iterations=5)
            # cv2.imshow('eroded image', eroded_img)
            # cv2.waitKey()

            # edges = cv2.Canny(filtered, 0, 0.5 * otsu_th)
            # eroded_img = cv2.dilate(edges, None, iterations=4)

            # padded_img = np.pad(eroded_img, (15, 15), mode='constant', constant_values=(1, 1))

            contours_img = np.copy(rgb_frame)
            contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            outer_contour_index = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
            print(outer_contour_index)

            print(f'found {len(contours)} contours')

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])
                bboxes_json[frame_count].append(boundRect[i])

            bboxes_img = np.copy(rgb_frame)
            for i in range(len(contours)):
                # tengo solo i contours figli dell'outer contour
                # print(f'contour {i}, hierarchy record {hierarchy[0][i]}')
                # if hierarchy[0][i][3] in outer_contour_index:
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                color = (255, 0, 0)
                cv2.drawContours(contours_img, contours_poly, i, color, thickness=3)
                cv2.rectangle(bboxes_img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                            (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 3)

            if debug:
                fig, ax = plt.subplots(2, 2, figsize=(14, 14))

                fig.canvas.set_window_title(f'Frame {frame_count}')

                # ax[0, 0].imshow(edges, cmap='gray')
                # ax[0, 0].set_title('canny')
                # ax[0, 1].imshow(filtered, cmap='gray')
                # ax[0, 1].set_title('blurred image')
                # ax[0, 2].imshow(eroded_img, cmap='gray')
                # ax[0, 2].set_title('eroded img')
                # ax[1, 0].imshow(th_img, cmap='gray')
                # ax[1, 0].set_title('otsu thresholding')
                # ax[1, 1].imshow(contours_img)
                # ax[1, 1].set_title('contours')
                # ax[1, 2].imshow(bboxes_img)
                # ax[1, 2].set_title('bounding boxes')

                ax[0, 0].imshow(frame, cmap='gray')
                ax[0, 0].set_title(f'frame')
                ax[0, 1].imshow(dilated_img, cmap='gray')
                ax[0, 1].set_title(f'dilated image')
                ax[1, 0].imshow(labeled_img)
                ax[1, 0].set_title(f'labeled image')
                ax[1, 1].imshow(bboxes_img)
                ax[1, 1].set_title(f'bounding boxes')

                plt.tight_layout()

                # plt.imshow(rgb_frame)
                # plt.gcf().canvas.mpl_connect('key_press_event', close)

                plt.show()
            else:
                out = cv2.cvtColor(bboxes_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'Output', out)
                cv2.waitKey(75)
        if not ret:
            break

    cap.release()
    #if debug or not noshow:
    cv2.destroyAllWindows()

    print(f'Writing bounding boxes to {output_file}')
    print(json.dumps(bboxes_json))
    with open(output_file, 'w') as file:
        json.dump(bboxes_json, file)


if __name__ == '__main__':
    args = parser.parse_args()

    main(**args.__dict__)
