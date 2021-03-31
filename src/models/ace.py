import os
from models.template import EnhancementStage
from models.color_balance import simplest_color_balance_per_channel
import cv2 as cv

class ACE(EnhancementStage):
    def __init__(self, slope, bright_clip, dark_clip):
        self.slope = slope
        self.bright_clip = bright_clip
        self.dark_clip = dark_clip
    
    def apply(self, input_path, output_path):
        ace(input_path, output_path, self.slope)
        output = cv.imread(output_path)
        output = self._color_adjustment(output,                                        
                                        self.bright_clip,
                                        self.dark_clip)
        cv.imwrite(output_path, output)
        
    @staticmethod
    def _color_adjustment(output, bright_clip, dark_clip):
        output = simplest_color_balance_per_channel(output, bright_clip, dark_clip)
        return output


def ace(input_path, output_path, a):
    os.system(f'./src/models/ace -a {a} -w 1/r -m interp:12 {input_path} {output_path}')
