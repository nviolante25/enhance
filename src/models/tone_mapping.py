from models.template import EnhancementStage
from models.color_balance import minmax_color_balance
import cv2 as cv
import numpy as np


class ToneMapping(EnhancementStage):
    """
    Tone mapping algorithms included in OpenCV, including Reinhard, Drago, and Mantiuk.
    For implementation details, please refer to https://docs.opencv.org/3.4/d6/df5/group__photo__hdr.html

    Refs:
        Reinhard et al. 'Photographic Tone Reproduction for Digital Images'
        http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf

        Mantiuk et al. 'A Perceptual Framework for Contrast Processing of High Dynamic Range Images'
        https://people.mpi-inf.mpg.de/~mantiuk/papers/mantiuk06ContrastProcTAP.pdf

        Drago et al. 'Adaptive Logarithmic Mapping For Displaying High Contrast Scenes'
        http://resources.mpi-inf.mpg.de/tmo/logmap/logmap.pdf

    """
    
    def __init__(self, tone_map, **args):
        if tone_map == 'tone_map':
            self.tone_map = cv.createTonemap(**args)
        elif tone_map == 'drago':
            self.tone_map = cv.createTonemapDrago(**args)
        elif tone_map == 'mantiuk':
            self.tone_map = cv.createTonemapMantiuk(**args)
        elif tone_map == 'reinhard':
            self.tone_map = cv.createTonemapReinhard(**args)
        else:
            raise ValueError(f'{tone_map} is not a valid Tone Mapping algorithm')
        
    @classmethod
    def create(cls, args):
        return cls(**args)
    
    def apply(self, image, output_path=None):
        output = self.tone_map.process(image.astype(np.float32))
        output[np.isnan(output)] = 0
        output = minmax_color_balance(output)
        if output_path:
            cv.imwrite(output_path, output)
        return output
