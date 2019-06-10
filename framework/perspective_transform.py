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


def determine_new_corners(corners, image, mode="zhang"):
    corners = order_points(corners)

    top_left, top_right, down_right, down_left = corners
    left_height = np.abs(top_left[1] - down_left[1])
    right_height = np.abs(top_right[1] - down_right[1])
    down_width = np.abs(down_left[0] - down_right[0])
    top_width = np.abs(top_right[0] - top_left[0])

    target_height = None
    target_width = None

    while not target_height or not target_width:

        if mode == "naive" or mode == "all":
            # New height and width will be max vertical and horizontal sides
            print("mode == naive")
            target_width = int(max(top_width, down_width))
            target_height = int(max(left_height, right_height))

            print(target_height / target_width)

        if mode == "habr" or mode == "all":
            # See https://habr.com/ru/post/223507/ for general idea of this approach
            print("mode == habr")
            mass_center = np.mean(corners, axis=0)

            def line(p1, p2):
                A = (p1[1] - p2[1])
                B = (p2[0] - p1[0])
                C = (p1[0] * p2[1] - p2[0] * p1[1])
                return A, B, -C

            def intersection(L1, L2):
                D = L1[0] * L2[1] - L1[1] * L2[0]
                Dx = L1[2] * L2[1] - L1[1] * L2[2]
                Dy = L1[0] * L2[2] - L1[2] * L2[0]
                if D != 0:
                    x = Dx / D
                    y = Dy / D
                    return x, y
                else:
                    assert(False, "There is no intersection between two lines")

            intersection_point = np.array(intersection(line(corners[0], corners[2]),
                                                       line(corners[1], corners[3])))
            delta = np.abs(mass_center - intersection_point)
            target_width = int(0.5 * (top_width + down_width) + 2 * delta[0])
            target_height = int(0.5 * (left_height + right_height) + 2 * delta[1])

            print(target_height / target_width)

        if mode == "zhang" or mode == "all":
                print("mode == zhang")
                m1x, m1y = down_left
                m2x, m2y = down_right
                m3x, m3y = top_left
                m4x, m4y = top_right

                # u0, v0 = down_left  ## it is not good way!!!

                v0 = image.shape[0] / 2
                u0 = image.shape[1] / 2

                m1x -= u0
                m1y -= v0
                m2x -= u0
                m2y -= v0
                m3x -= u0
                m3y -= v0
                m4x -= u0
                m4y -= v0

                k2 = ((m1y - m4y) * m3x - (m1x - m4x) * m3y + m1x * m4y - m1y * m4x) / \
                     ((m2y - m4y) * m3x - (m2x - m4x) * m3y + m2x * m4y - m2y * m4x)
                k3 = ((m1y - m4y) * m2x - (m1x - m4x) * m2y + m1x * m4y - m1y * m4x) / \
                     ((m3y - m4y) * m2x - (m3x - m4x) * m2y + m3x * m4y - m3y * m4x)

                print(k2, k3)

                f_squared = -((k3 * m3y - m1y) * (k2 * m2y - m1y) + (k3 * m3x - m1x) * (k2 * m2x - m1x)) / \
                            ((k3 - 1) * (k2 - 1))

                def sqr(x):
                    return x ** 2

                wh_ratio = np.sqrt(
                    (sqr(k2 - 1) + sqr(k2 * m2y - m1y) / f_squared + sqr(k2 * m2x - m1x) / f_squared) /
                    (sqr(k3 - 1) + sqr(k3 * m3y - m1y) / f_squared + sqr(k3 * m3x - m1x) / f_squared)
                )

                print("hight / width:", 1 / wh_ratio)
                print("width / hight:", wh_ratio)

                target_width = max(top_width, down_width)
                target_height = max(left_height, right_height)
                target_height = max(target_height, target_width / wh_ratio)
                target_width = target_height * wh_ratio

                if np.isnan(target_width):
                    target_width = None
                    mode = "naive"

    new_corners = np.array([[0, 0],
                            [target_width, target_height],
                            [target_width, 0],
                            [0, target_height]])
    return order_points(new_corners)


def remove_perspective_distortion(image, corners):
    new_corners = determine_new_corners(corners, image)
    transformed_image = transform_four_points_to_four_points(image, corners, new_corners)
    return transformed_image



