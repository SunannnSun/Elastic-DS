import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.damm.damm_class import damm_class
from src.ds_opt.dsopt_class import dsopt_class


# load data
input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. DAMM demo data
4. DEMO
Enter the corresponding option number: '''
input_opt  = input(input_message)

x, x_dot, x_att, x_init = load_tools.load_data(int(input_opt))

dim = x.shape[1]

param ={
    "mu_0":           np.zeros((dim, )), 
    "sigma_0":        0.1 * np.eye(dim),
    "nu_0":           dim,
    "kappa_0":        0.1,
    "sigma_dir_0":    0.1,
    "min_thold":      10
}

damm  = damm_class(x, x_dot, param)
gamma = damm.begin()


from lpv_ds_a.utils_ds.rearrange_clusters import rearrange_clusters
# gmm_struct = rearrange_clusters(damm.Priors, damm.Mu, damm.Sigma, x_att.T)
assignment_arr = np.argmax(gamma, axis=0)

plot_tools.plot_gmm(x, assignment_arr)
# plt.show()


from elastic_gmm.gaussian_kinematics import create_transform_azi


geo_test_arr = [
    [(0.1167, 0.4660, -np.pi/4), (0.8718, 0.4733, np.pi/4)],
    [(0.1167, 0.2660, -np.pi/4), (0.8718, 0.6733, np.pi/4)],
    [(0.2, 0.85, -np.pi/4), (0.7, 0.35, np.pi/4)]
]
geo_config = geo_test_arr[0]
O_s = np.array([create_transform_azi(np.array(geo_config[0][:2]), geo_config[0][2]), 
                create_transform_azi(np.array(geo_config[1][:2]), geo_config[1][2])])





a = 1