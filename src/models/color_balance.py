import numpy as np


def simplest_color_balance(image, high_th, low_th, new_max=255, dtype=np.uint8, new_min=0):
    high_value = np.percentile(image, high_th)
    low_value = np.percentile(image, low_th)
    image = np.maximum(np.minimum(image, high_value), low_value)
    image = ((new_max - new_min) / (high_value - low_value) * (image - low_value) + new_min)

    return image.astype(dtype)


def simplest_color_balance_per_channel(image, bright_clip, dark_clip, new_max=255, dtype=np.uint8, new_min=0):
    output = np.zeros_like(image, dtype=dtype)
    for i in [0,1,2]:
        output[:,:,i] = simplest_color_balance(image[:,:,i], bright_clip, dark_clip, new_max, dtype, new_min)
    return output


def minmax_color_balance(image, new_max=255, dtype=np.uint8, new_min=0):
    high_value = np.max(image)
    low_value = np.min(image)
    image = ((new_max - new_min) / (high_value - low_value) * (image - low_value) + new_min)
    return image.astype(dtype)

