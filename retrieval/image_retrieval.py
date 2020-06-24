import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import csv
import json


PAINTINGS_CSV = 'data/data.csv'
# using Hamming distance
NORM = cv2.NORM_HAMMING
BF = cv2.BFMatcher(NORM, crossCheck=True)
ORB = cv2.ORB_create(500, 1.3)


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

    def get_json(self):
        return {
            'filename': self.filename,
            'title': self.title,
            'author': self.author,
            'room': self.room
        }


def compute_kp_descr(im, orb):
    # find keypoints and descriptors from an image given a detector
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_im, None)
    return keypoints, descriptors


def get_paintings(orb, db):
    """
    compute the descriptors of all the image of the db with the orb detector
    :param orb: orb detector
    :return: list of Image objects
    """
    paintings = []
    with open(PAINTINGS_CSV, 'r') as f:
        f_reader = csv.DictReader(f)
        for row in f_reader:
            im = cv2.imread(f"{db}/{row['Image']}")
            kp, descr = compute_kp_descr(im, orb)
            image = Image(filename=row['Image'], image=im, descriptors=descr, keypoints=kp)
            image.title = row['Title']
            image.author = row['Author']
            image.room = row['Room']
            paintings.append(image)
    return paintings


def nearest_neighbors(query_image, reference_paintings, n_paintings):
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
    for image in reference_paintings:
        # brute force match
        matches = BF.match(image.descriptors, query_image.descriptors)
        # save the sorted matches matches
        image.matches = sorted(matches, key = lambda x: x.distance)
        paintings.append(image)
    # compute for each set of matches the avg 'divergence' to the query one
    avg_distances = [(np.mean([match.distance for match in p.matches]), p) for p in paintings]
    avg_distances = sorted(avg_distances, key=lambda x: x[0])
    top_results = [x[1] for x in avg_distances[0:n_paintings]]
    # update best matches on query image
    matches = BF.match(query_image.descriptors, top_results[0].descriptors)
    query_image.matches = sorted(matches, key=lambda x: x.distance)
    # return a list of paintings sorted by distance from the query one
    return [x[1] for x in avg_distances[0:n_paintings]]


def draw_matches(query_image, images_matched, num_kp_matched):
    """
    Plot both the most similar image found in the db with the corresponding
    keypoints matching and all the 10 most similar images
    :param query_image:
    :param images_matched:
    :param num_kp_matched: number of best matches to show
    """

    # Show top num_kp_matched matches
    plt.title(f'Best match: {images_matched[0].title}')
    plt.axis('off')
    """
    cv::drawMatches (
    InputArray img1,
    const std::vector< KeyPoint > &keypoints1,
    InputArray img2,
    const std::vector< KeyPoint > &keypoints2,
    const std::vector< DMatch > &matches1to2,
    InputOutputArray outImg,
    const Scalar &matchColor=Scalar::all(-1),
    const Scalar &singlePointColor=Scalar::all(-1),
    const std::vector< char > &matchesMask=std::vector< char >(),
    DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT)
    """

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
        plt.axis('off')
        plt.imshow(images_matched[i].image)
    plt.show()


def check_paths(source, findings):
    if not os.path.exists(source):
        print('no source of images')
        exit(1)
    if args.save_findings:
        if not os.path.exists(findings):
            os.mkdir(findings)


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Painting Retrieval module')

    parser.add_argument("--database", dest='paintings', help="PaintingsDB / Directory containing all the painting",
                        default="data/paintings_db", type=str)
    parser.add_argument("--source", dest='source', help="Source of the query images",
                        default="data/Rectified", type=str)
    parser.add_argument("--findings", dest='findings', help="Image-matchings / "
                                                            "Directory to store the frame with the corresponding "
                                                            "sorted paintings retrieved",
                        default="data/findings", type=str)
    parser.add_argument("--save", dest="save_findings", help="Do you want to save the results?", default=True)
    parser.add_argument("--show", dest="show", help="Do you want to show the results?", default=False)
    parser.add_argument("--n_paintings", dest="n_paintings", help="Number of retrieved paintings saved", default=10)
    parser.add_argument("--n_matches", dest="n_matches", help="Number of matches between the query image and the best "
                                                              "image retrieved shown", default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    check_paths(args.source, args.findings)
    # those parameters are optimal for this purpose
    print('--- getting descriptors of each painting in the db... ---')
    start = time.time()
    paintings = get_paintings(ORB, args.paintings)
    end = time.time()
    print(f'operation took: {end - start}s')
    for filename in os.listdir(args.source):
        if filename.endswith('png'):
            print(f'--- retrieve the painting {filename}... ---')
            start = time.time()
            im = cv2.imread(f'{args.source}/{filename}')
            kp, descr = compute_kp_descr(im, ORB)
            query_image = Image('query_image', im, descr, kp)
            # findings is an array of Images
            findings = nearest_neighbors(query_image, paintings, args.n_matches)
            end = time.time()
            print(f'operation took: {end - start}s')
            if args.save_findings:
                finding = {}
                finding['frame'] = filename
                json_findings = [f.get_json() for f in findings]
                finding['paintings'] = json_findings
                with open(f'{args.findings}/findings.json', 'w+') as outfile:
                    json.dump(finding, outfile)
            if args.show:
                draw_matches(query_image, findings, args.n_matches)

