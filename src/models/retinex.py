import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from models.color_balance import simplest_color_balance, simplest_color_balance_per_channel, minmax_color_balance
from models.template import EnhancementStage
from numpy.fft import fft2, ifft2

class MultiScaleRetinexHSV(EnhancementStage):
    """
    Local contrast enhancement method based on Retinex Theory and applied to the V component of the HSV input image.
    In addition, in order to account for color problems, a blend between the input image (with histogram stretching)
    and the contrast enhanced output is performed.

    Ref:
        Petro et al. 'Multiscale Retinex'
        https://www.ipol.im/pub/art/2014/107/
    """
    def __init__(self, scales = [450, 650], bright_clip=99.9, dark_clip=0.1, color_weight=0.7, output_weight=0.3):
        super().__init__()
        self.scales = scales
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip
        self.color_weight = color_weight
        self.output_weight = output_weight
        
    @classmethod
    def create(cls, args):
        return cls(**args)
        
    def apply(self, input, output_path=None):
        msr = multi_scale_retinex_hsv(input, self.scales)
        output = self._color_adjustment(input, 
                                        msr, 
                                        self.bright_clip,
                                        self.dark_clip,
                                        self.color_weight,
                                        self.output_weight)
        if output_path:
            cv.imwrite(output_path, output)
        return output
        
    @staticmethod
    def _color_adjustment(input, output, bright_clip, dark_clip, color_weight, output_weight):
        color = simplest_color_balance_per_channel(input, bright_clip, dark_clip)
        output = cv.addWeighted(color, color_weight, output, output_weight, 0)
        return output
    

def single_scale_retinex(input, scale):
    input = input.astype(np.float32)
    kernel = cv.getGaussianKernel(int(4*scale), scale)
    kernel = kernel@kernel.T
    blured_input = np.abs(ifft2(fft2(input)*fft2(kernel, input.shape)))
    out = np.log10(1.0 + input) - np.log10(1.0 + blured_input)
    return out


def multi_scale_retinex(input, scales):
    output = np.zeros(input.shape, dtype=np.float32)
    for scale in scales:
        output += single_scale_retinex(input, scale)
    output /= len(scales)
    return output


def multi_scale_retinex_hsv2(image, scales=[15., 80., 150.], high_bright_clip=99.99, low_bright_clip=99.0, dark_clip=1, 
                            mode='linear', color_weight=0.6, contrast_weight=0.4):
    intensity_channel_contrast = image.mean(axis=2, dtype=np.float32) 
    intensity_msr = multi_scale_retinex(intensity_channel_contrast, scales)
    intensity_msr = simplest_color_balance(intensity_msr, high_bright_clip, dark_clip, new_max=255, dtype=np.uint8)
    
    # Substitue HSV from log image
    intensity_channel_color = image_color.mean(axis=2, dtype=np.float32).astype(np.uint8) 
    output_hsv = cv.cvtColor(image_color, cv.COLOR_BGR2HSV)
    if mode == 'linear':
        output_hsv[:,:,2] = cv.addWeighted(intensity_channel_color, color_weight, intensity_msr, contrast_weight, 0)
    elif mode == 'power':
        output_hsv[:,:,2] = 255 * ((intensity_channel_color/255)**color_weight * (intensity_msr/255)**contrast_weight)
    else:
        raise ValueError(f'Mode {mode} is not valid. Available modes are: linear and power')
    output = cv.cvtColor(output_hsv, cv.COLOR_HSV2BGR)
    return output


def multi_scale_retinex_hsv(image, scales):
    # Apply Retinex
    intensity_channel = image.mean(axis=2, dtype=np.float32)
    msr = multi_scale_retinex(intensity_channel, scales)
    msr = simplest_color_balance(msr, 99.9, 0.1, 255, np.uint8)
    
    # Substitue V channel for Retinex output
    output_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    output_hsv[:,:,2] = msr
    output = cv.cvtColor(output_hsv, cv.COLOR_HSV2BGR)
    return output
