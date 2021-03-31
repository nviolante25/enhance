import os
from glob import glob
from shutil import copyfile
import numpy as np

if __name__ == '__main__':
    data_dir = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples'
    outputs_dir = '/home/nviolante/projects/retinex/outputs'
    random_order = True
    scene_dirs = glob(data_dir+'/*')
    if not scene_dirs:
        raise ValueError(f"Error: no scenes found at {data_dir}")
    
    
    scenes = [os.path.basename(scene_dir) for scene_dir in scene_dirs]
    
    output_dir = '/home/nviolante/projects/retinex/outputs/evaluation3'
    
    referece_file_path = os.path.join(output_dir, 'reference.txt')
    for scene in scenes:
        os.makedirs(os.path.join(output_dir, scene), exist_ok=True)
        
        images_to_compare = { 
            'quicklook_ps_ev': glob(os.path.join(data_dir, scene, '*Quicklook_PS_LS_2_99_EV.tif'))[0],
            'retinex_hsv_02': glob(os.path.join(outputs_dir, 'retinex_hsv', 'version_02', scene, 'after*.tif'))[0],
            'reinhard': glob(os.path.join(outputs_dir, 'histogram_stretching', 'version_00', scene, 'after*.tif'))[0],
            'reinhard_11': glob(os.path.join(outputs_dir, 'best_reinhard', 'version_11', scene, 'after*.tif'))[0],
            'reinhard_12': glob(os.path.join(outputs_dir, 'best_reinhard', 'version_12', scene, 'after*.tif'))[0],
            'bertalmio_08': glob(os.path.join(outputs_dir, 'bertalmio', 'version_08', scene, 'after*.tif'))[0],
            # 'ps_nuestro': glob(os.path.join(outputs_dir, 'histogram_stretching', 'version_09', scene, 'after*.tif'))[0],
            # 'log_stretch': glob(os.path.join(outputs_dir, 'log_stretch', 'version_00', scene, 'after*'))[0],
            # 'ace_00': glob(os.path.join(outputs_dir, 'ACE', 'version_00', scene, '*Full.tif'))[0],
            # 'original': glob(os.path.join(data_dir, scene, '*Full.tif'))[0],
            # 'quicklook': glob(os.path.join(data_dir, scene, '*Quicklook.tif'))[0],
            # 'quicklook_v2': glob(os.path.join(data_dir, scene, '*Quicklook_v2.tif'))[0],
            # 'quicklook_ps': glob(os.path.join(data_dir, scene, '*Quicklook_PS_LS_2_99.tif'))[0],
            # 'retinex_hsv_05': glob(os.path.join(outputs_dir, 'retinex_hsv', 'version_05', scene, 'after*'))[0],
            # 'ace_01': glob(os.path.join(outputs_dir, 'ACE', 'version_01', scene, '*Full.tif'))[0],
        }
        if random_order:
            keys = np.random.permutation(list(images_to_compare.keys()))
        else:
            keys = list(images_to_compare.keys())
            
            
        with open(referece_file_path, 'a') as f:
            f.write(scene + '\n')
        for i, key in enumerate(keys):
            src_path = images_to_compare[key]
            
            if random_order:
                dst_path = os.path.join(output_dir, scene, f'test_{i}.tif')
            else:
                dst_path = os.path.join(output_dir, scene, f'{key}.tif')
                
            copyfile(src_path, dst_path)
            
            with open(referece_file_path, 'a') as f:
                f.write(f'\t{key}: test_{i} \n')
        
        
        
        
    
    
    print()