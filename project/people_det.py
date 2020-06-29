from __future__ import division

from torch.autograd import Variable
from project.people_det_config.darknet import Darknet
from project.people_det_config.util import *
import random
import pickle as pkl
import argparse
import os


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon",
                        default="data/video_frames/prova2.mp4", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions",
                        default=0.8)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.5)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="people_det_config/cfg/yolov3-c1.cfg", type=str)
    parser.add_argument("--classes", dest='classes_file', help="Classes file",
                        default="people_det_config/coco.names", type=str)
    parser.add_argument("--pallete", dest='pallete_file', help="Pallete file",
                        default="people_det_config/pallete", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="people_det_config/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="256", type=str)
    parser.add_argument('--onein', help='Evaluate one frame every N frames.',
                        type=int, default=10)
    return parser.parse_args()


class PeopleDetector:
    num_classes = 1
    bbox_attrs = 5 + num_classes
    CUDA = torch.cuda.is_available()

    def __init__(self, config_dir,
                 confidence=0.8, nms_thesh=0.5, reso=256):

        print("Checking config dir")
        self.__check_config_dir__(config_dir)
        print("Config dir correct")

        print("Loading network")

        cfgfile = f"{config_dir}/cfg/yolov3-c1.cfg"
        weightsfile = f"{config_dir}/yolov3.weights"
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)

        classes = f"{config_dir}/coco.names"
        pallete = f"{config_dir}/pallete"
        self.classes = load_classes(classes)
        self.colors = pkl.load(open(pallete, "rb"))

        self.confidence = confidence
        self.nms_thesh = nms_thesh
        print("Network loaded")
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])
        if self.CUDA:
            self.model.cuda()
        self.model.eval()

    def __check_config_dir__(self, config_dir):
        """
        check if the configuration diractory contains all the necessary files
        :param config_dir:
        :return:
        """
        if not os.path.exists(config_dir):
            print(f'dir: {config_dir} does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/cfg/yolov3-c1.cfg'):
            print(f'dir: {config_dir}/cfg/yolov3-c1.cfg does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/coco.names'):
            print(f'dir: {config_dir}/coco.names does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/darknet.py'):
            print(f'dir: {config_dir}/darknet.py does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/pallete'):
            print(f'dir: {config_dir}/pallete does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/util.py'):
            print(f'dir: {config_dir}/util.py does not exist')
            exit(1)
        if not os.path.exists(f'{config_dir}/yolov3.weights'):
            print(f'dir: {config_dir}/yolov3.weights does not exist')
            exit(1)

    def __prepare_input__(self, img, inp_dim):
        """
        Prepare image for inputting to the neural network.
        Perform tranpose and return Tensor
        """
        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (custom_resize(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim

    def write(self, x, img):
        """
        it writes the bounding boxes on the image for visualization
        :param x: b boxes
        :param img:
        :return: img with bounding boxes
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        # if a class which is not a person is found do nothing
        if cls == 0:
            label = "{0}".format(self.classes[cls])
            color = random.choice(self.colors)
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    def detect(self, frame):
        """
        it detects all the people in the given frame
        :param frame:
        :return: the found bounding boxes, the
        """

        img, orig_im, dim = self.__prepare_input__(frame, pd.inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if pd.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            prediction = self.model(Variable(img), pd.CUDA)
        output = write_results(prediction, self.confidence, self.num_classes, nms_conf=self.nms_thesh)

        if type(output) == int:
            return None

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(pd.inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (pd.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (pd.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        return output


if __name__ == '__main__':

    args = arg_parse()

    pd = PeopleDetector(config_dir='people_det_config')

    videofile = args.video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    frame_count = 0
    one_in = args.onein

    while cap.isOpened():
        frame_count += 1

        ret, frame = cap.read()

        height, width, layers = frame.shape
        new_h = int(height / 2)
        new_w = int(width / 2)
        frame = cv2.resize(frame, (new_w, new_h))

        if ret:
            if frame_count % one_in == 0:
                output = pd.detect(frame)

                if output is None:
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('x'):
                        break
                    continue

                list(map(lambda x: pd.write(x, frame), output))

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('x'):
                    break
        else:
            break




