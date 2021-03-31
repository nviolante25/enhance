import os
from glob import glob
import ast

def get_version_params(info_path):
    with open(info_path) as f:
        params = ast.literal_eval(f.readlines()[2].strip('\n'))
    return params

if __name__ == '__main__':
    experiment_dir = '/home/nviolante/projects/retinex/outputs/reinhard_exp'
    info_per_version_path = sorted(glob(os.path.join(experiment_dir, 'version*', 'info.txt')))

    params = [get_version_params(info_path) for info_path in info_per_version_path]
    print()