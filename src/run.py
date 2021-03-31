from glob import glob
import cv2 as cv
import os
from time import time
from utils.read_tif import read_bgrnir_tif_as_bgr
import matplotlib.pyplot as plt
import numpy as np
from models.retinex import MultiScaleRetinexHSV
from models.ace import ACE
from models.tone_mapping import ToneMapping
from models.photo_stretching import PhotoStretching
from models.histogram_stretching import HistogramStretching, HistogramStretchingHSV, HistogramStretchingYUV, HistogramStretchingHSV2
from models.bertalmio import Bertalmio
from utils.experiment_version import experiment_version_dir
from utils.histograms import bgr_histograms


def run(cfg, scene_dirs, output_dir, keep_intermediate=False, save_histograms=False):
    name = cfg['name']
    light_adaptation = cfg['light_adaptation']['model'].create(cfg['light_adaptation']['args'])
    contrast_enhance = cfg['contrast_enhance']['model'].create(cfg['contrast_enhance']['args'])

    mean_time = 0
    total_images = 0

    for scene_dir in scene_dirs:
        scene = os.path.basename(scene_dir)
        print(f'Scene: {scene}')
        os.makedirs(os.path.join(output_dir, scene), exist_ok=True)
        if save_histograms:
            histograms_dir = os.path.join(output_dir, scene, 'histograms')
            os.makedirs(histograms_dir, exist_ok=True)

        image_paths = glob(scene_dir + '/*Full.tif')
        for i, image_path in enumerate(image_paths):
            image = read_bgrnir_tif_as_bgr(image_path)
            print(f'\tProcessing image: {i} of size {image.shape}')

            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, scene, f'after_{filename}')
            intermediate_path = os.path.join(output_dir, scene, 'intermediate.tif')
            tic = time()

            intermediate_image = light_adaptation.apply(image, intermediate_path)
            if name == 'ace':
                contrast_enhance.apply(intermediate_path, output_path)
            else:
                contrast_enhance.apply(intermediate_image, output_path)
            toc = time()

            if not keep_intermediate:
                os.remove(intermediate_path)

            if save_histograms:
                fig = bgr_histograms(image, cumulative=True)
                fig.savefig(os.path.join(histograms_dir, 'original.png'))
                fig = bgr_histograms(intermediate_image, cumulative=True)
                fig.savefig(os.path.join(histograms_dir, 'intermediate.png'))
                fig = bgr_histograms(cv.imread(output_path), cumulative=True)
                fig.savefig(os.path.join(histograms_dir, 'final.png'))

            print(f'\tTime: {toc - tic:.2f} seconds')

            mean_time += toc - tic
            total_images += 1

    mean_time /= total_images
    time_info = f'Processed {total_images} images. Average time was {mean_time:.2f} seconds'
    print(time_info)
    info_path = os.path.join(output_dir, 'info.txt')
    with open(info_path, 'w') as f:
        f.write(name + '\n')
        f.write(cfg['light_adaptation']['model'].__name__ + '\n')
        f.write(str(cfg['light_adaptation']['args']) + '\n')
        f.write(cfg['contrast_enhance']['model'].__name__ + '\n')
        f.write(str(cfg['contrast_enhance']['args']) + '\n')
        f.write(time_info)


if __name__ == '__main__':
    from itertools import product

    gammas = [0.8]
    intensities = [-0.5, 0]
    color_adapts = [1.0]
    light_adapts = [0]
    dark_clips = [0]
    tests = [{'name': 'stretch_hsv',
              'light_adaptation': {'model': HistogramStretchingHSV2,
                                   'args': {'bright_clip': 99,
                                            'dark_clip': dark_clip,
                                            'new_min': 0
                                            }
                                    },
              'contrast_enhance': {'model': ToneMapping,
                                   'args': {'tone_map': 'reinhard',
                                            'gamma': gamma,
                                            'intensity': intensity,
                                            'color_adapt': color_adapt,
                                            'light_adapt': light_adapt
                                            }
                                   },
              }
             for gamma, intensity, color_adapt, light_adapt,dark_clip in product(gammas, intensities, color_adapts, light_adapts, dark_clips)]
    # tests = [{'name': 'nico-tests',
    #         'light_adaptation': {'model': HistogramStretchingHSV,
    #                 'args': {'bright_clip': 99.99,
    #                         'dark_clip': 0.0
    #                         }
    #                 },
    #           'contrast_enhance': {'model': ToneMapping,
    #                                'args': {'tone_map': 'reinhard',
    #                                         'intensity': -1.5,
    #                                         'light_adapt': 1.0,
    #                                         'color_adapt': 1.0,
    #                                         'gamma': 1.0
    #                                         }
    #                                },
    #           }
    # ]
    data_dir = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples'
    scene_dirs = glob(data_dir + '/*')
    if not scene_dirs:
        raise ValueError(f"Error: no scenes found at {data_dir}")

    for cfg in tests:
        output_dir = f'/home/nviolante/projects/retinex/outputs/{cfg["name"]}'
        output_dir = experiment_version_dir(output_dir)
        run(cfg, scene_dirs, output_dir, keep_intermediate=False, save_histograms=True)
