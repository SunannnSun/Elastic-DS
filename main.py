import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, process_tools, plot_tools
from src.damm.damm_class import damm_class
from src.dsopt.dsopt_class import dsopt_class


# Load data (Optional)
p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=0, num_traj=1, sub_sample=1)
# p_raw, q_raw, t_raw = load_tools.load_demo_dataset()

# Process data (Optional)
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)

# """
dim = p_in.shape[1]
param ={
    "mu_0":           np.zeros((dim, )), 
    "sigma_0":        0.1 * np.eye(dim),
    "nu_0":           dim,
    "kappa_0":        0.1,
    "sigma_dir_0":    0.1,
    "min_thold":      10
}
damm  = damm_class(p_in, p_out, param)
gamma = damm.begin()
assignment_arr = np.argmax(gamma, axis=0)


from lpv_ds_a.utils_ds.rearrange_clusters import rearrange_clusters
gmm_struct = rearrange_clusters(damm.Priors, damm.Mu.T, damm.Sigma, p_att.T)


from elastic_gmm.gaussian_kinematics import create_transform
T_init = create_transform(p_init[0], q_init[0])
T_att  = create_transform(p_att, q_att)


from elastic_gmm.generate_transfer import start_adapting
p_io = [np.hstack((p_in, p_out)).T]
traj_data, gmm_struct, old_joints, new_joints = start_adapting(p_io, gmm_struct, T_init, T_att)


plot_tools.plot_ds(p_in ,traj_data[0].T, old_joints, new_joints, assignment_arr, T_init)
plt.show()
