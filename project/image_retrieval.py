import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import csv
import json


def check_paths(source, findings):
    if not os.path.exists(source):
        print('no source of images')
        exit(1)
    if args.save_findings:
        if not os.path.exists(findings):
            os.mkdir(findings)


class Image:
    def __init__(self, filename, image, descriptors, keypoints):
        self.filename = filename
        self.image = image
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.matches = None
        self.distance = 0   # similarity distance from the frame
        self.title = ''
        self.author = ''
        self.room = ''

    def get_json(self):
        return {
            'filename': self.filename,
            'title': self.title,
            'author': self.author,
            'room': self.room
        }


def adjust_image(img, contrast_factor, brightness_factor=0):
    """
    Adjust contrast and brightness of an Image.
    :param img: numpy ndarray to be adjusted.
    :param contrast_factor: How much to adjust the contrast.
    :param brightness_factor: How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    :return: numpy ndarray: Contrast adjusted image.
    """

    table = np.array([ i*contrast_factor + brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)


def draw_matches(query_image, images_matched, num_kp_matched, accurate):
    """
    Plot both the most similar image found in the db with the corresponding
    keypoints matching and all the 10 most similar images
    :param query_image:
    :param images_matched:
    :param num_kp_matched: number of best matches to show
    """

    # Show top num_kp_matched matches
    correct = 'Correct Match' if accurate else 'Wrong Match'
    title_obj = plt.title(f'Best match: {images_matched[0].title} -> {correct}')
    plt.setp(title_obj, color='g') if accurate else plt.setp(title_obj, color='r')
    print(f'Best match: {images_matched[0].title}, distance: {images_matched[0].distance}')
    print(f'accurate: {accurate}')
    print(f'matches: {[m.distance for m in images_matched[0].matches]}')
    plt.axis('off')

    img_matches = cv2.drawMatches(query_image.image, query_image.keypoints, images_matched[0].image,
                                  images_matched[0].keypoints,
                                  query_image.matches[:num_kp_matched],
                                  images_matched[0].image, flags=2)
    plt.imshow(img_matches)
    plt.show()
    fig = plt.figure(figsize=(6, 6))
    plt.title('First 10 matches')
    plt.axis('off')
    for i, im in enumerate(images_matched):
        fig.add_subplot(5, 2, i + 1)
        plt.title(f"#{i+1}: {im.title}")
        plt.setp(title_obj, color='g') if accurate else plt.setp(title_obj, color='r')
        print(f"#{i+1}: {im.title}, distance: {im.distance}")
        plt.axis('off')
        plt.imshow(images_matched[i].image)
    plt.show()


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Painting Retrieval module')

    parser.add_argument("--database", dest='paintings', help="PaintingsDB / Directory containing all the painting",
                        default="data/paintings_db", type=str)
    parser.add_argument("--source", dest='source', help="Source of the query images",
                        default="data/Rectified-video", type=str)
    parser.add_argument("--findings", dest='findings', help="Image-matchings / "
                                                            "Directory to store the frame with the corresponding "
                                                            "sorted paintings retrieved",
                        default="data/findings", type=str)
    parser.add_argument("--save", dest="save_findings", help="Do you want to save the results?", default=False)
    parser.add_argument("--show", dest="show", help="Do you want to show the results?", default=True)
    parser.add_argument("--n_paintings", dest="n_paintings", help="Number of retrieved paintings saved", default=10)
    parser.add_argument("--n_matches", dest="n_matches", help="Number of matches between the query image and the best "
                                                              "image retrieved shown", default=10)
    return parser.parse_args()


class Retrieval:
    def __init__(self, paintings_db, paintings_csv):
        # using Hamming distance
        self.BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ORB = cv2.ORB_create(500, 1.25)
        self.paintings_csv = paintings_csv
        self.paintings = self.__get_paintings__(paintings_db)

    def __check_kp__(self, kp, min_bound, max_bound):
        new_kp = []
        for k in kp:
            if min_bound[0] < k.pt[0] < max_bound[0] and min_bound[1] < k.pt[1] < max_bound[1]:
                new_kp.append(k)
        return new_kp

    def __compute_kp_descr__(self, im):
        # find keypoints and descriptors from an image given a detector
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        (X, Y) = gray_im.shape
        kp = self.ORB.detect(gray_im, None)
        kp = self.__check_kp__(kp, (X/4, Y/4), (3*X/4, 3*Y/4))
        kp, des = self.ORB.compute(gray_im, kp)
        return kp, des

    def __get_paintings__(self, db):
        """
        compute the descriptors of all the image of the db with the orb detector
        :param orb: orb detector
        :return: list of Image objects
        """
        paintings = []
        with open(self.paintings_csv, 'r') as f:
            f_reader = csv.DictReader(f)
            for row in f_reader:
                im = cv2.imread(f"{db}/{row['Image']}")
                kp, descr = self.__compute_kp_descr__(im)
                image = Image(filename=row['Image'], image=im, descriptors=descr, keypoints=kp)
                image.title = row['Title']
                image.author = row['Author']
                image.room = row['Room']
                paintings.append(image)
        return paintings

    def __nearest_neighbors__(self, query_image, reference_paintings, n_paintings):
        """
        Given a list of descriptors <reference_descriptors>, find the <match_num>
        images that have the most similar descriptors to the <image_descriptors>.

        :param query_image: the image we want to retrieve
        :param reference_paintings: array of Images representing the paintings of the db
        :param n_paintings: number of retrieved paintings we want to return
        :return: array of Image objects. They are sorted from the most to the less
                 similar to the <image_descriptors>
        """

        paintings = []
        # find best matches for each reference image
        for painting in reference_paintings:
            # brute force match
            matches = self.BF.match(painting.descriptors, query_image.descriptors)
            # save the sorted matches matches
            painting.matches = sorted(matches, key=lambda x: x.distance)
            paintings.append(painting)
        # compute for each set of matches the avg 'divergence' to the query one
        for p in paintings:
            p.distance = np.mean([match.distance for match in p.matches])
        sorted_paintings = sorted(paintings, key=lambda x: x.distance)
        # update best matches on query image
        matches = self.BF.match(query_image.descriptors, sorted_paintings[0].descriptors)
        query_image.matches = sorted(matches, key=lambda x: x.distance)
        # return a list of paintings sorted by distance from the query one
        return sorted_paintings[0:n_paintings]

    def __is_accurate__(self, paintings):
        distances = [p.distance for p in paintings[:10]]
        dist_best_one = [m.distance for m in paintings[0].matches]

        if len(dist_best_one)<3:
            return False
        if dist_best_one[2] <= 30:
            if paintings[0].title != "Ritratto d'uomo":
                if distances[0] > 65:
                    return False
                else:
                    return True
            else:
                if distances[9] - distances[0] < 7:
                    return False
                else:
                    return True
        else:
            return False

    def retrieve_image(self, image, n_matches=10, save_findings=False, show=False, findings_dir='data/findings'):
        start = time.time()

        im_adj = adjust_image(image, 2)
        # compute the descr and the kp of the image and save it as Image class
        kp, descr = self.__compute_kp_descr__(im_adj)
        if not kp or len(kp) < 10:
            return None

        query_image = Image('query_image', image, descr, kp)

        print(f'--- retrieve the painting {query_image.filename}... ---')

        # do the retrieval
        findings = self.__nearest_neighbors__(query_image, self.paintings, n_matches)
        if findings:
            accurate = self.__is_accurate__(findings)
        else:
            accurate = False

        end = time.time()
        print(f'operation took: {end - start}s')
        if save_findings:
            finding = {}
            finding['frame'] = filename
            json_findings = [f.get_json() for f in findings]
            finding['paintings'] = json_findings
            finding['accurate'] = accurate
            with open(f'{findings_dir}/findings.json', 'w+') as outfile:
                json.dump(finding, outfile)
        if show and accurate:
            draw_matches(query_image, findings, n_matches, accurate)

        if accurate:
            return [f.get_json() for f in findings]
        else:
            return None


if __name__ == '__main__':

    args = arg_parse()
    check_paths(args.source, args.findings)

    # those parameters are optimal for this purpose
    print('--- getting descriptors of each painting in the db... ---')
    start = time.time()

    ret = Retrieval(args.paintings)

    end = time.time()
    print(f'operation took: {end - start}s')

    for filename in os.listdir(args.source):
        im = cv2.imread(f'{args.source}/{filename}')
        ret.retrieve_image(im, show=True)

