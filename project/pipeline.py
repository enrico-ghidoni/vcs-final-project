import argparse
import cv2
import numpy as np
import detection as detection
import perspective_correction as perspective_correction
import image_retrieval as image_retrieval
import people_det as people_detection
import os
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--video',
                    help='Absolute path for video file to process.',
                    default='../data/Video Images/prova4.mp4')
parser.add_argument('--paintings-db',
                    help='Absolute path for paintings db directory.',
                    default='../data/paintings_db')
parser.add_argument('--paintings-csv',
                    help='Absolute path for data.csv file.')
parser.add_argument('--config-people-detection',
                    help='Absolute path for people detection config dir.',
                    default='../project/people_det_config')
parser.add_argument('--output-path',
                    help='Absolute path to store pipeline process results.',
                    default='findings')
parser.add_argument('--onein',
                    help='Evaluate one frame every N frames in the video.',
                    type=int,
                    default=10)
parser.add_argument('--debug',
                    help='Show detailed comparison of image processing.',
                    action='store_true')
parser.add_argument('--silent',
                    help=' Do not show image results while processing.',
                    action='store_true')


class Pipeline(object):
    def __init__(self, video, paintings_db, paintings_csv, output_path, config_people_detection,
                 onein):
        self._video = video
        self._output_path = output_path
        self._onein = onein

        self._detection = detection.PaintingDetection()
        self._rectification = perspective_correction.PaintingRectification()
        self._retrieval = image_retrieval.Retrieval(paintings_db, paintings_csv)
        self._people_det = people_detection.PeopleDetector(config_people_detection)

        # create output path if doesn't already exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self._detection_out = os.path.join(output_path, 'detection.json')
        self._rectification_out = os.path.join(output_path, 'rectification.json')
        self._retrieval_out = os.path.join(output_path, 'retrieval.json')
        self._video_out = os.path.join(output_path, 'video.avi')

        self._frame_w = None
        self._frame_h = None

    def start(self):
        self._cap = None
        self._cur_frame = 1

        self._bounding_boxes = {}
        self._ims_rectified = {}
        self._ims_match = {}
        self._ppl_bounding_boxes = {}
        self._frames_to_save = []

        cap = cv2.VideoCapture(self._video)
        if cap.isOpened():
            self._cap = cap
        else:
            raise Exception(f'Unable to start video stream for {self._video}')

    def next_frame(self, auto_skip = True):
        self.frame_bounding_boxes = None
        self.frame_ims_rectified = None
        self.frame_ims_matches = None

        if auto_skip:
            if self._cur_frame % self._onein == 0:
                ret, frame = self._read_next_frame()
            else:
                while self._cur_frame % self._onein != 0:
                    ret, frame = self._read_next_frame()
        else:
            ret, frame = self._read_next_frame()

        if not ret:
            return False
        if ret and self._cur_frame % self._onein == 0:
            if self._frame_w is None and self._frame_h is None:
                self._frame_w = frame.shape[1]
                self._frame_h = frame.shape[0]
            self.frame = frame
            self.frame_bounding_boxes = self._detection.detect_paintings(frame)
            self.frame_ims_rectified = self._rectification.perspective_correction(frame, self.frame_bounding_boxes)
            self.frame_ims_matches = []
            for im_rectified in self.frame_ims_rectified:
                if im_rectified is not None:
                    self.frame_ims_matches.append(self._retrieval.retrieve_image(im_rectified))
                else:
                    self.frame_ims_matches.append(None)

            self.frame_ppl_bounding_boxes = self._people_det.detect(frame)

            self._bounding_boxes[self._cur_frame] = self.frame_bounding_boxes
            self._ims_rectified[self._cur_frame] = self.frame_ims_rectified
            self._ims_match[self._cur_frame] = self.frame_ims_matches
            self._frames_to_save.append(self.image_matching_bounding)

        return True

    def stop(self):
        self._cap.release()

    def save_outputs(self):
        with open(self._detection_out, 'w') as detection_out:
            json.dump(self._bounding_boxes, detection_out)
        with open(self._retrieval_out, 'w') as retrieval_out:
            json.dump(self._ims_match, retrieval_out)

        for frame in self._ims_rectified.keys():
            suffix = f'f_{frame}'
            for i, image in enumerate(self._ims_rectified[frame]):
                if image is not None:
                    filename = f'{suffix}_{i}_rect.png'
                    path = os.path.join(self._output_path, filename)
                    print(path)
                    cv2.imwrite(path, image)

        video_writer = cv2.VideoWriter(self._video_out, cv2.VideoWriter_fourcc(*'XVID'), 30, (self._frame_w, self._frame_h))
        for frame in self._frames_to_save:
            print(video_writer.write(frame))

    def autoplay(self, save_outputs = True):
        self.start()
        while self.next_frame():
            pass
        self.stop()
        self.save_outputs()

    @property
    def image_matching_bounding(self):
        image = np.copy(self.frame)
        thickness = 3
        if self.frame_bounding_boxes:
            for i, box in enumerate(self.frame_bounding_boxes):
                if self.frame_ims_matches[i] and self.frame_ims_matches[i] is not None:
                    color = (0, 255, 0)
                    text = self.frame_ims_matches[i][0]['title']
                else:
                    color = (0, 0 , 255)
                    text = 'No match'

                x, y, w, h = box
                ulx, uly = x, y
                brx, bry = x + w, y + h
                label_point = (ulx + thickness, uly + h - thickness)

                cv2.rectangle(image, (ulx, uly), (brx, bry), color, thickness)
                cv2.putText(image, text, label_point, cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        if self.frame_ppl_bounding_boxes is not None:
            color = (255, 0, 0)
            for tensor in self.frame_ppl_bounding_boxes:
                self._people_det.write(tensor, image)

        return image

    def _read_next_frame(self):
        self._cur_frame += 1
        return self._cap.read()


def console_entry_point():
    args = parser.parse_args()

    pipe = Pipeline(**args.__dict__)
    pipe.autoplay()

if __name__ == '__main__':
    console_entry_point()
