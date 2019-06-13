from matplotlib import pyplot as plt
import cv2
import numpy as np


FILE_NUM_TO_SAVE = 0
MAX_SIZE_TO_SHOW = 960


def visualize_image(image, title="", mode="both", inplace=False, restart=False):
    """
    Function for visualizing image
    :param image: image, which was read by opencv
    :param title: title for the window, optional
    :param mode: string from list ["show", "save", "both"]
    :param inplace: boolean
    """
    if restart:
        global FILE_NUM_TO_SAVE
        FILE_NUM_TO_SAVE = 0

    if inplace:  # matplotlib mode
        plt.figure(figsize=(20, 20))
        plt.title(title)
        image_to_show = image
        if len(image.shape) > 2:
            assert(image.shape[2] == 3)
            image_to_show = image[:, :, ::-1]
            plt.imshow(image_to_show)
        else:
            assert(len(image.shape) == 2)
            plt.imshow(image_to_show, cmap='gray')

    else:
        image_to_show = image

        if mode == "save" or mode == "both":  # saving to file
            cv2.imwrite(str(FILE_NUM_TO_SAVE) + ".jpg", image_to_show)
            FILE_NUM_TO_SAVE += 1

        if mode == "show" or mode == "both":  # show, using OpenCV windows
            if max(image.shape) > MAX_SIZE_TO_SHOW:
                max_side = np.max(image_to_show.shape[:2])
                new_max_side = MAX_SIZE_TO_SHOW
                ratio = max_side / new_max_side
                new_size = tuple(map(lambda x: int(x), image_to_show.shape[:2][::-1] / ratio))
                image_to_show = cv2.resize(image_to_show, new_size,
                                           interpolation=cv2.INTER_CUBIC)
                title += "Real size is {} larger".format(round(ratio, 2))

            cv2.imshow(title, image_to_show)
            cv2.waitKey(0)
