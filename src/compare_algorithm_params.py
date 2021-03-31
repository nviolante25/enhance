import cv2 as cv
from glob import glob
import os
from ast import literal_eval

class Viewer:
    def __init__(self, outputs_dir):
        self.version_dirs = glob(os.path.join(outputs_dir, 'version*'))
        self.scenes = list(map(lambda x: x.split('/')[-2], glob(os.path.join(self.version_dirs[0], '*/'))))
        info_paths = [glob(os.path.join(version_dir, 'info.txt'))[0]
                      for version_dir in self.version_dirs]
        
        self.params = [self._get_version_params(info_path) for info_path in info_paths]
        
    @staticmethod
    def _get_version_params(info_path):
        with open(info_path) as f:
            params = literal_eval(f.readlines()[4])
        return params
    
    
    def _get_desired_int(self, scene, desired_intensity):
        idx = next(idx for idx, param in enumerate(self.params) if param['intensity'] == desired_intensity)
        return glob(os.path.join(self.version_dirs[idx], scene, '*.tif'))[0]
    
    def show_image_with(self, scene, intensity):
        image = cv.imread(self._get_desired_int(scene, intensity))
        cv.imshow('hola', image)

    def show(self):
        title_window = cv.namedWindow('hola', cv.WINDOW_NORMAL)
        cv.resizeWindow(title_window, 1000, 1000)
        
        cv.createTrackbar('inte', 'hola' , 0, 2, lambda x:self.show_image_with(self.scenes[1], x))

    
if __name__ == '__main__':
    outputs_dir = './outputs/nico-tests'
    v = Viewer(outputs_dir)
    print()  
    v.show()

    cv.waitKey(0)
    cv.destroyAllWindows()
    
    print()
        
        
    


