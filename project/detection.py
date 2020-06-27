import cv2
import numpy as np


class PaintingDetection(object):
    def __init__(self):
        self.bounding_boxes = None

    def detect_paintings(self, image):
        """Entry point for painting detection.

        Args:
            image: opencv-python image in BGR color space on which paintings are detected

        Returns:
            list: list of bounding boxes, each bounding box is a [x, y, w, h] list where x and y are the coordinates of the top-left corner of the box, w and h are the box width and height respectively
        """
        self._original_image = image

        im_grey = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        im_filtered = self._gaussian_smoothing(im_grey, adaptive_kernel= False)
        _, im_thold = cv2.threshold(im_filtered, 180, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        im_dilated = cv2.dilate(im_thold, None, iterations = 4)

        contours, _ = self._find_contours(im_dilated, keep_outer = True)
        bounding_boxes = self._build_bounding_boxes(contours, mean_filter = True)

        return bounding_boxes

    def _gaussian_smoothing(self, image, adaptive_kernel = False):
        """Apply Gaussian Blur filter to `image`.

        Args:
            image: opencv-python image in grey color space
            adaptive_kernel (boolean, optional): if `True` the kernel size is inferred from the image. Defaults to `False`.

        Returns:
            image: smoothed opencv-python image
        """
        if adaptive_kernel:
            std = np.std(image)
            single_k_size = int(np.ceil(3 * std) // 2 * 2 + 1)
            k_size = (single_k_size, single_k_size)
        else:
            k_size = (5, 5)

        return cv2.GaussianBlur(image, k_size, 0)

    def _find_contours(self, image, keep_outer = True):
        """Find contours in `image`.

        Args:
            image: opencv-python image in grey color space
            keep_outer (boolean, optional): if `True` only outer contours are returned. Defaults to `True`.

        Returns:
            tuple: a tuple containing:
                contours (list): a list containing contours (same as `cv2.findContours()`)
                hierarchy (list): a list describing contour hierarchy (same as `cv2.findContours()`)
        """
        im_contours = np.copy(image)

        contours, hierarchy = cv2.findContours(im_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if keep_outer:
            outer_contours_index = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
            contours = [contours[i] for i in outer_contours_index]
            hierarchy = [hierarchy[0][i] for i in outer_contours_index]

        return contours, hierarchy

    def _build_bounding_boxes(self, contours, mean_filter = True):
        """Build bounding boxes from contours.

        Args:
            contours (tuple): tuple of contours in the same format as return value of `cv2.findContours()`
            mean_filter (bool, optional): return only boxes with area above the mean area of all boxes built. Defaults to `True`.

        Returns:
            list: list of bounding boxes, each bounding box is a [x, y, w, h] list where x and y are the coordinates of the top-left corner of the box, w and h are the box width and height respectively
        """
        contours_poly = [None] * len(contours)
        boxes = [None] * len(contours)
        tot_area = 0
        for i, contour in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
            boxes[i] = cv2.boundingRect(contours_poly[i])
            tot_area += boxes[i][2] * boxes[i][3]

        if mean_filter and len(contours) > 0:
            mean_area = tot_area / len(contours)
            boxes = [box for box in boxes if (box[2] * box[3]) >= mean_area]

        return boxes

