import os
from glob import glob


def experiment_version_dir(output_dir):
    version_dirs = sorted(glob(output_dir+'/version*'))
    if version_dirs:
        current_version_dir = os.path.basename(version_dirs[-1])
        next_version = int(current_version_dir.split('_')[-1]) + 1
    else:
        next_version = 0

    next_version_dir = os.path.join(output_dir, f'version_{str(next_version).zfill(2)}')
    return next_version_dir
