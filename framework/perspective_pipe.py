import cv2
import sys
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import cm
sys.path.append("..")
from framework import visualize, perspective_transform, edges_detection


def process_file(full_file_path, verbose=0):
    # reading image
    initial_image = cv2.imread(full_file_path)
    if verbose >= 2:
        visualize.visualize_image(initial_image, restart=True)
    else:
        visualize.visualize_image(initial_image, mode="save", restart=True)

    # reducing size
    resized_image, resizing_ratio = edges_detection.reduce_image_size(initial_image)
    if verbose >= 4:
        visualize.visualize_image(resized_image)
    else:
        visualize.visualize_image(resized_image, mode="save")

    # median filter
    smoothed_image = edges_detection.blur_image(resized_image, cv2.medianBlur, 5, 3)
    if verbose >= 4:
        visualize.visualize_image(smoothed_image)
    else:
        visualize.visualize_image(smoothed_image, mode="save")

    # edges detection
    edges = edges_detection.canny_edge_detector(smoothed_image)
    if verbose >= 3:
        visualize.visualize_image(edges)
    else:
        visualize.visualize_image(edges, mode="save")

    # hough transform
    hough_lines = edges_detection.find_hough_lines(edges, mode="skimage")

    # show lines
    cdst = edges_detection.draw_lines_on_edges(edges, hough_lines)
    if verbose >= 4:
        visualize.visualize_image(cdst)
    else:
        visualize.visualize_image(cdst, mode="save")

    # save hough space
    if verbose >= 1:
        h, theta, d = hough_line(edges)

        plt.figure(figsize=(6, 6))
        plt.imshow(np.log(1 + h),
                   extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[-1], d[0]],
                   cmap=cm.gray, aspect=1 / 5)
        # plt.title('Hough transform')
        plt.xlabel('Angles (degrees)')
        plt.ylabel('Distance (pixels)')
        plt.tight_layout()
        plt.savefig("hough_transform.jpg", bbox_inches='tight')

        for j, color in zip(range(len(hough_lines)), ["r", (127 / 255, 255 / 255, 0)]):
            for i in range(0, len(hough_lines[j])):
                dist, angle = hough_lines[j][i][0]
                print(angle / np.pi * 180, dist)
                plt.plot(np.rad2deg(angle), dist, color=color, marker="o")

        plt.savefig("hough_transform_with_points.jpg", bbox_inches='tight')

    # if bad lines were found
    if len(hough_lines) < 2 or min(len(hough_lines[0]), len(hough_lines[1])) < 2:
        return

    # find lines intersections
    intersections = edges_detection.segmented_intersections(hough_lines)
    for point in intersections:
        cv2.circle(cdst, point, 3, (0, 255, 255), -1)
    if verbose >= 4:
        visualize.visualize_image(cdst)
    else:
        visualize.visualize_image(cdst, mode="save")

    # find corners
    corners = KMeans(n_clusters=4).fit(np.array(intersections)).cluster_centers_
    for corner in corners:
        corner = tuple(np.array(corner, dtype=int))
        cv2.circle(cdst, corner, 3, (255, 255, 0), -1)

    if verbose >= 3:
        visualize.visualize_image(cdst)
    else:
        visualize.visualize_image(cdst, mode="save")

    # correct image
    corners = np.array(corners * resizing_ratio, dtype=int)
    restored = perspective_transform.remove_perspective_distortion(initial_image, corners)
    if verbose >= 2:
        visualize.visualize_image(restored)
    else:
        visualize.visualize_image(restored, mode="save")

    print(restored.shape, restored.shape[0] / restored.shape[1])



