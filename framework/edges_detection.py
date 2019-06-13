import cv2
import numpy as np
from collections import defaultdict
from skimage.transform import hough_line, hough_line_peaks


import sys
sys.path.append("..")
from framework import visualize


SHORT_SIDE_SMALL_LEN = 300
CANNY_EDGE_DETECTOR_PARAMETER = 1.5


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
    dispersion = grey_dispersion = np.std(grey_image)
    string = "greyscale image"
    # visualize.visualize_image(grey_image, string)
    resulted_channel = grey_image
    for i in range(3):
        single_channel = image[:, :, i]
        new_dispersion = np.std(single_channel)
        new_string = f"{i} channel of image"
        # visualize.visualize_image(single_channel, new_string)
        if new_dispersion > CANNY_EDGE_DETECTOR_PARAMETER * grey_dispersion and new_dispersion > dispersion:
            dispersion = new_dispersion
            string = new_string
            resulted_channel = grey_image

    print(f"For CannyEdgeDetection {string} was chosen")

    high_thresh, thresh_im = cv2.threshold(resulted_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    # edges = cv2.Canny(image, low_thresh, high_thresh)
    edges = cv2.Canny(resulted_channel, low_thresh, high_thresh)
    return edges


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def find_hough_lines(edges, threshold="adaptive", mode="skimage"):
    """

    :param edges:
    :param threshold:
    :param mode: which mode to use (["opencv", "skimage"])
    :return: two groups of Hough lines (horizontal and vertical)
    """
    if mode == "opencv":
        if type(threshold) != str:
            hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        else:
            left_bound, right_bound = 0, 255
            left_number, right_number = np.infty, 0
            while True:
                t = (left_bound + right_bound) // 2
                hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, t)
                new_len = 0 if hough_lines is None else len(hough_lines)
                if new_len >= 2:
                    first_group, second_group = segment_by_angle_kmeans(hough_lines)
                    min_len = min(len(first_group), len(second_group))
                else:
                    min_len = 0
                if min_len < 4:
                    right_bound = t
                    right_number = min_len
                else:
                    left_bound = t
                    left_number = min_len
                if min_len == 2 or right_bound - left_bound <= 1:
                    break

        hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, t)
        if hough_lines is None or len(hough_lines) < 4:
            print("Failed to find good Hough lines")

    else:
        h, theta, d = hough_line(edges)
        peaks = hough_line_peaks(h, theta, d)
        print("Number of detected lines ", len(peaks[0]))
        if len(peaks[0]) < 4:
            print("Failed to find good Hough lines")
        hough_lines = [[[dist, angle]] for _, angle, dist in zip(*peaks)]

    segmented_lines = segment_by_angle_kmeans(hough_lines)
    return segmented_lines


def draw_lines_on_edges(edges, two_lines_groups):
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Draw the lines
    for j, color in zip(range(len(two_lines_groups)), [(0, 0, 255), (127, 255, 0)]):
        for i in range(0, len(two_lines_groups[j])):
            rho = two_lines_groups[j][i][0][0]
            theta = two_lines_groups[j][i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(result, pt1, pt2, color, 1, cv2.LINE_AA)

    return result


def intersection_of_lines(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection_of_lines(line1, line2))

    return intersections

