import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)


from src.util import load_tools, process_tools, plot_tools
from src.se3_class import se3_class

from src.generate_transfer import start_adapting


'''Load data'''
p_raw, q_raw, t_raw, dt = load_tools.load_UMI()


'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)

p_in = p_in[:100]
p_out = p_out[:100]
q_in = q_in[:100]
q_out = q_out[:100]

data = np.hstack((p_in, p_out))


'''Plot demo'''
# plot_tools.plot_demo(p_in, q_in)


'''Run lpvds'''
se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
se3_obj.begin()


'''GMM Structure object'''
from src.util.rearrange_clusters import rearrange_clusters
Priors_list = se3_obj.gmm.Prior
Mu_list = se3_obj.gmm.Mu
Sigma_list = se3_obj.gmm.Sigma

K = int(len(Priors_list) / 2)
N = 3

Priors = np.zeros((K))
Mu = np.zeros((N, K))
Sigma = np.zeros((K, N, N))

for k in range(K):
    Priors[k] = Priors_list[k] * 2
    Mu[:, k] = Mu_list[k][0]
    Sigma[k, :, :] = Sigma_list[k][:N, :N]

att = p_att
old_gmm_struct = rearrange_clusters(Priors, Mu, Sigma, att)


'''Define geometric constraints'''

v1 = p_in[25] - p_in[0]
v1 /= np.linalg.norm(v1)
# v1 = np.array([0, 0, 1])
v2 = np.cross(v1, np.array([0, 1, 0]))
v3 = np.cross(v1, v2)

R_s = np.column_stack((v1, v2, v3))

u1 = p_in[-1] - p_in[-25]
u1 /= np.linalg.norm(u1)
# u1 = np.array([0, 0, -1])
u2 = np.cross(u1, np.array([0, 1, 0]))
u3 = np.cross(u1, u2)

R_e = np.column_stack((u1, u2, u3))

T_s = np.zeros((4, 4))
T_e = np.zeros((4, 4))


# T_s[:3, :3] = q_in[0].as_matrix()
T_s[:3, :3] = R_s
T_s[:3, -1] = p_in[0]
T_s[-1, -1] = 1

# T_e[:3, :3] = q_in[-1].as_matrix()
T_e[:3, :3] = R_e
T_e[:3, -1] = p_in[-1]
T_e[-1, -1] = 1


'''Start adapting'''
traj_data, gmm_struct, old_anchor, new_anchor = start_adapting([data.T], old_gmm_struct, T_s, T_e)




'''Record Relative Orientation'''
ori_sigma_old = [R.identity] * K
for k in range(K):

    Sigma_k = old_gmm_struct.Sigma[k, :, :]
    _, eigen_vectors = np.linalg.eig(Sigma_k)
    ori_sigma_old[k] = R.from_matrix(eigen_vectors)

    # ori_sigma_old[k] = R.from_matrix(old_gmm_struct.Sigma[k, :, :])


assignment_array = se3_obj.gmm.assignment_arr
M = assignment_array.shape[0]
relative_ori = [R.identity()] * M
for i in range(M):
    k = assignment_array[i]
    relative_ori[i] = q_in[i] * ori_sigma_old[k].inv() 


'''Update new orientation'''
ori_sigma_new = [R.identity] * K
for k in reversed(range(K)):
    Sigma_k = gmm_struct.Sigma[k, :, :]
    _, eigen_vectors = np.linalg.eig(Sigma_k)
    ori_sigma_new[k] = R.from_matrix(eigen_vectors)

    # ori_sigma_new[k] = R.from_matrix(gmm_struct.Sigma[k, :, :])


new_ori = [R.identity()] * M
for i in range(M):
    k = assignment_array[i]
    new_ori[i] = relative_ori[i] * ori_sigma_new[k]


plot_tools.demo_vs_adjust(p_in, traj_data[0][:3, :].T, old_anchor, new_anchor, q_in, new_ori)


# plot_tools.demo_vs_adjust_gmm(p_in, traj_data[0][:3, :].T, se3_obj.gmm, old_gmm_struct, new_ori, gmm_struct)

# plot_tools.plot_gmm(p_in, se3_obj.gmm)
plot_tools.plot_quat(q_in, new_ori)

plt.show()






