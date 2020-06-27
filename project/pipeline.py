import argparse
import cv2
import detection
import perspective_correction
import image_retrieval
import os
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--video',
                    help='Absolute path for video file to process.')
parser.add_argument('--paintings-db',
                    help='Absolute path for paintings db directory.')
parser.add_argument('--paintings-csv',
                    help='Absolute path for data.csv file.')
parser.add_argument('--output-path',
                    help='Absolute path to store pipeline process results.')
parser.add_argument('--onein',
                    help='Evaluate one frame every N frames in the video.',
                    type=int,
                    default=1)
parser.add_argument('--debug',
                    help='Show detailed comparison of image processing.',
                    action='store_true')
parser.add_argument('--silent',
                    help=' Do not show image results while processing.',
                    action='store_true')


def main(video, paintings_db, paintings_csv, output_path, onein, debug = False, silent = False):
    frame_count = 0
    bounding_boxes_acc = {}
    images_rectified = {}
    painting_detection = detection.PaintingDetection()
    persp = perspective_correction.PaintingRectification()
    retrieval = image_retrieval.Retrieval(paintings_db, paintings_csv)
    print(f'Pipeline starting with video {video}')

    Path(output_path).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if frame_count % onein == 0 and ret:
            print(f'Starting painting detection')
            bounding_boxes = painting_detection.detect_paintings(frame)
            ims_rectified = persp.perspective_correction(frame, bounding_boxes)
            for i, im_rectified in enumerate(ims_rectified):
                print(cv2.imwrite(f'{output_path}/{frame_count}-{i}.png', im_rectified))
                im_matches = retrieval.retrieve_image(im_rectified)
                print(im_matches)
            images_rectified[frame_count] = ims_rectified
            bounding_boxes_acc[frame_count] = bounding_boxes
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

    painting_detection_out = os.path.join(output_path, 'painting-detection.json')
    with open(painting_detection_out, 'w') as pd_out:
        json.dump(bounding_boxes_acc, pd_out)

def pipeline_entry_point():
    """Console entry point."""

    args = parser.parse_args()

    main(**args.__dict__)


if __name__ == '__main__':
    pipeline_entry_point()