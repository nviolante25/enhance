import numpy as np
from models.template import EnhancementStage
from models.color_balance import minmax_color_balance
import cv2 as cv

class PhotoStretching(EnhancementStage):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def apply(image, output_path=None):
        max_img = np.nanmax(image)
        mean_img = np.nanmean(image)
        lambda_coeff = (max_img - (2 * mean_img)) / (max_img * mean_img)
        output = np.divide(image, (1 + image * lambda_coeff))
        output = minmax_color_balance(output)
        if output_path:
            cv.imwrite(output_path, output)
        return output

