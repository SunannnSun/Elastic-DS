import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.damm.damm_class import damm_class
from src.ds_opt.dsopt_class import dsopt_class

from elastic_gmm.generate_transfer import start_adapting
from utils.pipeline_func import get_gmm, get_ds
from utils.plotting import plot_full_func


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

data = np.hstack((x, x_dot))

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
Priors_list = damm.Priors
Mu_list = damm.Mu
Sigma_list = damm.Sigma

K = int(len(Priors_list))
N = 2

Priors = np.zeros((K))
Mu = np.zeros((N, K))
Sigma = np.zeros((K, N, N))

for k in range(K):
    Priors[k] = Priors_list[k]
    Mu[:, k] = Mu_list[k]
    Sigma[k, :, :] = Sigma_list[k][:N, :N]

att = x_att[0]



'''Define geometric constraints'''

v1 = x[25] - x[0]
v1 /= np.linalg.norm(v1)
# v1 = np.array([0, 0, 1])
a, b = v1
v2 = np.array([-b, a])
R_s = np.column_stack((v1, v2))

u1 = x[-1] - x[-25]
u1 /= np.linalg.norm(u1)
# u1 = np.array([0, 0, -1])
a, b = u1
u2 = np.array([-b, a])
# u2 = np.cross(u1, np.array([0, 1]))
R_e = np.column_stack((u1, u2))

T_s = np.zeros((3, 3))
T_e = np.zeros((3, 3))

# T_s[:3, :3] = q_in[0].as_matrix()
T_s[:2, :2] = R_s
T_s[:2, -1] = x[0]
T_s[-1, -1] = 1

# T_e[:3, :3] = q_in[-1].as_matrix()
T_e[:2, :2] = R_e
T_e[:2, -1] = x[-1]
T_e[-1, -1] = 1



demo_geo = []
old_gmm_struct = rearrange_clusters(Priors, Mu, Sigma, att)
old_ds_struct = get_ds(old_gmm_struct,[data.T], None, demo_geo)



'''Start adapting'''
traj_data, gmm_struct, old_anchor, new_anchor = start_adapting([data.T], old_gmm_struct, T_s, T_e)

geo_config = demo_geo
ds_struct = get_ds(gmm_struct, traj_data,  None, geo_config)


plot_full_func(ds_struct, old_ds_struct)



