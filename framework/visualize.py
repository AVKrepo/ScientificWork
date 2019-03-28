from matplotlib import pyplot as plt
import cv2
import numpy as np


FILE_NUM_TO_SAVE = 0
MAX_SIZE_TO_SHOW = 960


def visualize_image(image, title="", mode="show", inplace=False):
    """
    Function for visualizing image
    :param image: image, which was read by opencv
    :param title: title for the window, optional
    :param mode: string from list ["show", "save"]
    :param inplace: boolean
    """
    if inplace:
        plt.figure(figsize=(20, 20))
        plt.title(title)
        image_to_show = image
        if len(image.shape) > 2:
            image_to_show = image[:, :, ::-1]
        plt.imshow(image_to_show)
    else:
        image_to_show = image
        if max(image.shape) > MAX_SIZE_TO_SHOW:
            max_side = np.max(image_to_show.shape[:2])
            new_max_side = MAX_SIZE_TO_SHOW
            ratio = max_side / new_max_side
            new_size = tuple(map(lambda x: int(x), image_to_show.shape[:2][::-1] / ratio))
            image_to_show = cv2.resize(image_to_show, new_size,
                                       interpolation=cv2.INTER_CUBIC)
            title += f"Real size is {round(ratio, 2)} larger"
        if mode == "show":
            cv2.imshow(title, image_to_show)
            cv2.waitKey(0)
        elif mode == "save":
            global FILE_NUM_TO_SAVE
            cv2.imwrite(str(FILE_NUM_TO_SAVE) + ".jpg", image_to_show)
            FILE_NUM_TO_SAVE += 1