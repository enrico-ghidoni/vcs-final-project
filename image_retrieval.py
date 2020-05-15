import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from descriptors import get_paintings, compute_kp_descr, Image


PAINTINGS_FOLDER = 'data/paintings_db'
QUERY_IMAGES = 'data/Rec_Image/Rectified'
# using Hamming distance
NORM = cv2.NORM_HAMMING
BF = cv2.BFMatcher(NORM, crossCheck=True)


def nearest_neighbors(image_descriptors, reference_descriptors, match_num=10):
    """
    Given a list of descriptors <reference_descriptors>, find the <match_num>
    images that have the most similar descriptors to the <image_descriptors>.

    :param image_descriptors: the descriptors of the image we want to retrieve
    :param reference_descriptors: array of tuples (<file_name>, <descriptors>)
    :param match_num: number of matches we want to return
    :return: array of Image objects. They are sorted from the most to the less
             similar to the <image_descriptors>
    """

    x = []
    # find best matches for each reference image
    for image in reference_descriptors:
        # brute force match
        matches = BF.match(image.descriptors, image_descriptors)
        # keep top 10 matches
        matches = sorted(matches, key = lambda x: x.distance)[:10]
        x.append([matches, image])
    # compute for each set of matches the avg 'divergence' to the query one
    avg_distances = [(np.mean([y.distance for y in match[0]]), match[1]) for match in x]
    avg_distances = sorted(avg_distances, key=lambda x: x[0])
    return [x[1] for x in avg_distances[0:match_num]]


def draw_matches(query_image, images_matched, num_kp_matched = 10):
    """
    Plot both the most similar image found in the db with the corresponding
    keypoints matching and all the 10 most similar images
    :param query_image:
    :param images_matched:
    :param num_kp_matched:
    """
    if len(images_matched) != 10:
        raise Exception('images matched are not 10')
    matches = BF.match(query_image.descriptors, images_matched[0].descriptors)
    matches = sorted(matches, key = lambda x: x.distance)
    # Show top 10 matches
    fi = plt.figure(figsize=(6, 6))
    plt.title(f'Best match: {images_matched[0].title}')
    plt.axis('off')
    img_matches = cv2.drawMatches(query_image.image, query_image.keypoints, images_matched[0].image,
                                  images_matched[0].keypoints, matches[:num_kp_matched],
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


if __name__ == '__main__':
    orb = cv2.ORB_create(500, 1.4, WTA_K=2)
    paintings_descriptors = get_paintings(orb)
    for filename in os.listdir(QUERY_IMAGES):
        if filename.endswith('png'):
            im = cv2.imread(f'{QUERY_IMAGES}/{filename}')
            kp, descr = compute_kp_descr(im, orb)
            query_image = Image('query_image', im, descr, kp)
            # findings is an array of tuples <distance_from_query_image, <image_name>
            findings = nearest_neighbors(query_image.descriptors, paintings_descriptors)
            draw_matches(query_image, findings)

