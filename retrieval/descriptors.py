import cv2
import csv
import datetime


PAINTINGS_FOLDER = 'paintings_db'
PAINTINGS_CSV = 'data.csv'
NORM = cv2.NORM_HAMMING


class Image:
    def __init__(self, filename, image, descriptors, keypoints):
        self.filename = filename
        self.image = image
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.matches = None
        self.title = ''
        self.author = ''
        self.room = ''


def compute_kp_descr(im, orb):
    # find keypoints and descriptors from an image given a detector
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_im, None)
    return keypoints, descriptors


def get_paintings(orb):
    """
    compute the descriptors of all the image of the db with the orb detector
    :param orb: orb detector
    :return: list of Image objects
    """
    paintings = []
    with open(PAINTINGS_CSV, 'r') as f:
        f_reader = csv.DictReader(f)
        for row in f_reader:
            im = cv2.imread(f"{PAINTINGS_FOLDER}/{row['Image']}")
            kp, descr = compute_kp_descr(im, orb)
            image = Image(filename=row['Image'], image=im, descriptors=descr, keypoints=kp)
            image.title = row['Title']
            image.author = row['Author']
            image.room = row['Room']
            paintings.append(image)
    return paintings

