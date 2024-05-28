#python3 -m task.main_start_end
import numpy as np
import os, sys
import argparse
from os.path import exists
import pickle
import matplotlib.pyplot as plt
import time


current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)



from elastic_gmm.split_traj import split_traj
from elastic_gmm.gaussian_kinematics import create_transform_azi
from elastic_gmm.generate_transfer import start_adapting
from utils.pipeline_func import get_gmm, get_ds
from utils.plotting import plot_full_func
from lpv_ds_a.utils_ds.structures import ds_gmms




def elastic_func(gmm, traj, start_constr, end_constr):
    """
    gmm: dictionary
    traj: 
    """


    new_traj, pi, mean_arr, cov_arr, _ = start_adapting(gmm, traj, start_constr, end_constr)
    new_gmm = ds_gmms()
    new_gmm.Mu = mean_arr.T
    new_gmm.Priors = pi
    new_gmm.Sigma = cov_arr

    return new_gmm, new_traj



if __name__ == "__main__":
    import pickle

    #----------------------------------------------------#
    #--------------------load data-----------------------#
    #----------------------------------------------------#
    """
    x:     an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations of N dimension,
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0, help='Choose Test Case')
    args = parser.parse_args()
    p = args.test

    task_name = 'startreach2d'
    task_idx  = 'all_2d'
    pkg_dir = os.getcwd()
    final_dir = pkg_dir + "/task_dataset/" + task_name + "/" + str(task_idx) + ".p"
    loaded_data = pickle.load(open(final_dir, 'rb'))
    demo_geo = loaded_data['geo']
    data = loaded_data['traj']
    data_drawn = data.copy()
    
    split_pts = np.array([])                       ##### need to delete later #####
    segment_traj = split_traj(data, split_pts, 2)  ##### need to delete later #####
    first_segment_data = segment_traj[0]           ##### need to delete later #####
    traj_batch = segment_traj[0]                   ##### need to delete later #####

    #----------------------------------------------------#
    #--------------------1st GMM and DS -----------------#
    #----------------------------------------------------#
    old_gmm_struct = get_gmm(first_segment_data)
    old_ds_struct = get_ds(old_gmm_struct, first_segment_data[0], None, demo_geo)


    #----------------------------------------------------#
    #------------------define constraints----------------#
    #----------------------------------------------------#
    geo_test_arr = [
        [(0.1167, 0.4660, -np.pi/4), (0.8718, 0.4733, np.pi/4)],
        [(0.1167, 0.2660, -np.pi/4), (0.8718, 0.6733, np.pi/4)],
        [(0.2, 0.85, -np.pi/4), (0.7, 0.35, np.pi/4)]
    ]
    geo_config = geo_test_arr[2]
    O_s = np.array([create_transform_azi(np.array(geo_config[0][:2]), geo_config[0][2]), 
                    create_transform_azi(np.array(geo_config[1][:2]), geo_config[1][2])])

    #----------------------------------------------------#
    #--------------------start adapting -----------------#
    #----------------------------------------------------#

    gmm_struct, traj_data = elastic_func(old_gmm_struct, first_segment_data[0], O_s[0], O_s[1])

    #----------------------------------------------------#
    #----------------------2nd DS -----------------------#
    #----------------------------------------------------#
    ds_struct = get_ds(gmm_struct, traj_data, None, geo_config)

    #----------------------------------------------------#
    #----------------------plot--------------------------#
    #----------------------------------------------------#
    plot_full_func(ds_struct, old_ds_struct)