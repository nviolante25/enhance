import cv2 as cv
from models.color_balance import simplest_color_balance_per_channel, simplest_color_balance, minmax_color_balance
from models.template import EnhancementStage
import numpy as np


def clip(input, min, max):
    return np.minimum(np.maximum(input, min), max)


class HistogramStretching(EnhancementStage):
    def __init__(self, bright_clip, dark_clip, new_max=255, new_min=0):
        super().__init__()
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip
        self.new_max = new_max
        self.new_min = new_min

    def apply(self, image, output_path=None):
        output = simplest_color_balance_per_channel(image, self.bright_clip, self.dark_clip, new_max=self.new_max,
                                                    new_min=self.new_min)
        if output_path:
            cv.imwrite(output_path, output)
        return output
    

class HistogramStretchingHSV(EnhancementStage):
    def __init__(self, bright_clip, dark_clip, new_max=255, new_min=0):
        super().__init__()
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip
        self.new_max = new_max
        self.new_min = new_min

    def apply(self, image, output_path=None):
        im_hsv=cv.cvtColor(np.float32(image),cv.COLOR_BGR2HSV)
        I = im_hsv[:,:,2]
        I_new = simplest_color_balance(I , self.bright_clip, self.dark_clip, I.max(), image.dtype)
        I_new = (I_new/I_new.max())*255
        output = np.zeros_like(image)
        I[I==0] = 1
        output[:,:,0] = image[:,:,0]*(I_new/I)
        output[:,:,1] = image[:,:,1]*(I_new/I)
        output[:,:,2] = image[:,:,2]*(I_new/I)
        if output_path:
            cv.imwrite(output_path, output)
        return output
    
    
class HistogramStretchingHSV2(EnhancementStage):
    def __init__(self, bright_clip, dark_clip, new_max=255, new_min=0):
        super().__init__()
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip
        self.new_max = new_max
        self.new_min = new_min

    def apply(self, image, output_path=None):
        im_hsv=cv.cvtColor(np.float32(image), cv.COLOR_BGR2HSV)
        I = im_hsv[:,:,2]
        I_new = simplest_color_balance(I , self.bright_clip, self.dark_clip, I.max(), np.float32)
        S = im_hsv[:,:,1]
        S_new = simplest_color_balance(S , self.bright_clip, self.dark_clip, S.max(), np.float32)
        
        im_hsv[:,:,1]= S_new
        im_hsv[:,:,2]= I_new
        
        output = cv.cvtColor(im_hsv, cv.COLOR_HSV2BGR).astype(image.dtype)
                
        if output_path:
            cv.imwrite(output_path, output)
        return output
    
    
class HistogramStretchingYUV(EnhancementStage):
    def __init__(self, bright_clip, dark_clip):
        super().__init__()
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip

    def apply(self, image, output_path=None):
        image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        Y = image_yuv[:,:,0].copy()
        
        new_Y = simplest_color_balance(Y, self.bright_clip, self.dark_clip, new_max=1.0, dtype=np.float32)
        image_yuv[:,:,0] = new_Y
        output = cv.cvtColor(image_yuv, cv.COLOR_YUV2BGR)
        if output_path:
            cv.imwrite(output_path, output)
        return output
