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
            gk_size = (5, 5)

            filtered = cv2.GaussianBlur(gray, gk_size, 0)
            otsu_th, th_img = cv2.threshold(filtered, 180, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

            dilated_img = cv2.dilate(th_img, None, iterations=4)

            contours_img = np.copy(rgb_frame)
            contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # keep only outer contours
            outer_contours_index = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
            contours = [contours[i] for i in outer_contours_index]
            hierarchy = [hierarchy[0][i] for i in outer_contours_index]
            print(f'outer contours indexes {outer_contours_index}')

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])

            # calculate the average area of the bounding boxes to remove the ones below
            bboxes_area = [box[2] * box[3] for box in boundRect]
            avg_area = np.mean(bboxes_area)
            print(f'average box area {avg_area}')
            boundRect = [box for box in boundRect if (box[2] * box[3]) >= avg_area]

            bboxes_img = np.copy(rgb_frame)
            for i, box in enumerate(boundRect):
                color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
                # cv2.drawContours(contours_img, contours_poly, i, color, thickness=3)
                cv2.rectangle(bboxes_img, (int(box[0]), int(box[1])), \
                            (int(box[0] + box[2]), int(box[1] + box[3])), color, 3)
                bboxes_json[frame_count].append(box)
            if not noshow:
                if debug:
                    fig, ax = plt.subplots(2, 2, figsize=(14, 14))

                    fig.canvas.set_window_title(f'Frame {frame_count}')

                    ax[0, 0].imshow(th_img, cmap='gray')
                    ax[0, 0].set_title(f'threshold')
                    ax[0, 1].imshow(dilated_img, cmap='gray')
                    ax[1, 0].set_title(f'labeled image')
                    ax[1, 1].imshow(bboxes_img)
                    ax[1, 1].set_title(f'bounding boxes')

                    plt.tight_layout()
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
