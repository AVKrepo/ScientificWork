import cv2
import numpy as np


SHORT_SIDE_SMALL_LEN = 300


def reduce_image_size(image, future_short_side_len=SHORT_SIDE_SMALL_LEN):
    initial_small_side_len = sorted(image.shape[:2])[0]
    resizing_ratio = initial_small_side_len / future_short_side_len
    resized_image = cv2.resize(image, None, fx=1 / resizing_ratio,
                               fy=1 / resizing_ratio, interpolation=cv2.INTER_CUBIC)
    return resized_image, resizing_ratio


def blur_image(image, blur, kernel_size, times=1):
    """
    Impose blur on image
    :param image: cv2 image
    :param blur: cv2 type of blur (for example: cv2.medianBlur or cv2.GaussianBlur)
    :param kernel_size: the same argument for blur
    :param times: optional, if is needed to impose blur several times
    :return: blurred image
    """
    resulted_image = image
    for i in range(times):
        try:
            # case of cv2.medianBlur
            resulted_image = blur(resulted_image, kernel_size)
        except:
            # case of cv2.GaussianBlur
            resulted_image = blur(resulted_image, (kernel_size, kernel_size), 0)
    return resulted_image


def canny_edge_detector(image):
    """

    :param image: colorful cv2 image
    :return: edges mask (np.uint8 array with values 0 or 255)
    """
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_thresh, thresh_im = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(image, low_thresh, high_thresh)
    return edges


def find_hough_lines(edges, threshold="adaptive"):
    """

    :param edges:
    :param threshold:
    :return:
    """
    if type(threshold) != str:
        hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    else:
        left_bound, right_bound = 0, 255
        left_number, right_number = np.infty, 0
        while True:
            t = (left_bound + right_bound) // 2
            hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, t)
            new_len = 0 if hough_lines is None else len(hough_lines)
            if new_len < 4:
                right_bound = t
                right_number = new_len
            else:
                left_bound = t
                left_number = new_len
            if new_len == 4 or right_bound - left_bound <= 1:
                break

    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, t)
    if hough_lines is None or len(hough_lines) < 4:
        print("Failed to find good Hough lines")
    return hough_lines

