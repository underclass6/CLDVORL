import numpy as np
from CLDV.replay_buffer import ReplayBuffer
from tqdm import tqdm

def filter_trajectories(env, source_buffer, dvrl, batch_size, device):
    trajRB = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=device)
    trajectories = []
    trajs_value = []
    trajs_size = source_buffer.trajs_size
    for i in tqdm(range(trajs_size)):
        traj = source_buffer.sample_trajectory(ind=i, to_device=False)
        trajRB.reset()
        for j in range(traj[0].shape[0]):
            trajRB.add(traj[0][j], traj[1][j], traj[2][j], traj[3][j], 1.0 - traj[4][j])
        dve_out, sel_vec = dvrl.data_valuate(trajRB, batch_size)
        dve_mean = np.mean(dve_out)
        print(dve_mean)
        if dve_mean < 0.0:
            continue
        trajectories.append(traj)
        trajs_value.append(dve_mean)
    return trajectories, trajs_value


def sort_trajectories(trajs_value):
    sorted_value_ind = sorted(range(len(trajs_value)), key=lambda k: trajs_value[k])
    return sorted_value_ind


def cldv_train(dvrl, rl_model, selected_source_buf, trajectories, sorted_value_ind):
    hop = int(len(sorted_value_ind) / 20)
    start_idx = 0
    for i in tqdm(range(0, 33 * hop, hop)):
        if i < 33 * hop / 4:
            start_idx = i  # after a certain number of training, the start position will remain constant

        ind = sorted_value_ind[start_idx:]

        selected_source_buf.reset()
        for index in ind:
            traj = trajectories[index]
            for j in range(traj[0].shape[0]):
                selected_source_buf.add(traj[0][j], traj[1][j], traj[2][j], traj[3][j], 1.0 - traj[4][j])
        dvrl.dict['training_max_timesteps'] = 1e4 * 3
        print(f'Size of new source buffer: {selected_source_buf.size}')
        dvrl.train_dvbca(selected_source_buf, m_type=rl_model)


def modify_rewards(source_buffer, dve_out, ratio=0.05):
    for i in tqdm(range(source_buffer.size)):
        source_buffer.reward[i] = (1-ratio) * source_buffer.reward[i] + ratio * dve_out[i]
    return source_buffer
