import numpy as np
import cv2


def order_points(points):
    """
    Sort points so that their order will be top-left, top-right, down-right, down-left.
    :param points: list of tuples / numpy array with shape (4, 2)
    :return: points (np.array), which are sorted by convention
    """
    points = np.array(points, dtype=np.float32)
    center_point = np.mean(points, axis=0)
    deltas = points - center_point
    imagine_numbers = deltas[:, 0] + 1.0j * deltas[:, 1]
    angles = 180 - np.angle(imagine_numbers, deg=True)
    new_order = np.argsort(angles)
    resulted_points = points[new_order]
    return resulted_points


def transform_four_points_to_four_points(image, start_points, end_points):
    """
    Perspective transform, using initial four points and resulted four points
    :param image:
    :param start_points: initial points of image corners
    :param end_points: resulted points of corners for image to be in the first coordinate quarter
    :return: transformed image
    """
    start_points = order_points(start_points)
    end_points = order_points(end_points)
    max_width = int(np.max(end_points[:, 0]) - 1)
    max_height = int(np.max(end_points[:, 1]) - 1)

    matrix = cv2.getPerspectiveTransform(start_points, end_points)
    resulted_image = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return resulted_image


