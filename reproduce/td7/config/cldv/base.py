from UtilsRL.misc import NameSpace

seed = 0
task = None
max_buffer_size = 2000000
discount = 0.99
max_action = 1.0

max_epoch = 1000
step_per_epoch = 1000
batch_size = 256

eval_interval = 10
eval_episode = 10
save_interval = 50
log_interval = 10
max_trajectory_length = 1000

name = "d4rl"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

actor_lr = critic_lr = encoder_lr = 3e-4

embedding_dim = 256
hidden_dim = 256

actor_update_interval = 2
target_update_interval = 250
policy_noise = 0.2
noise_clip = 0.5
lam = 0.1

use_checkpoint = False
use_lap_buffer = True

normalize_obs = False
normalize_reward = False

### DVRL params
env = task
dev = 'cuda'
shuffle = False
num_workers = 4
run_num = -1
ex_configs = None
dist_epsilon = 1.2
rl_model = 'BCQ'
rl_model_t = 'fresh'
baseline = 'static'
moving_average_type = 1
rl_model_tr_ev_t = 'CQL'
baseline_model = 'train'
lmbda = 0.5
split_ratio = 0.9


############## Delta Classfier ##############
updates_per_step_cla = 100
env_num = 0
isMediumExpert = False
dcla_cuda = True
dcla_lr = 0.0003
dcla_hidden_size = 256
dcla_epochs = 30
dcla_batch_size = 256
dcla_ratio = 0.95

############## Env and dataset related #############
d4rl_source_env = None
d4rl_target_env = None
env_source_level = 1
env_target_level = 3
source_env_friction = 1.0
source_env_mass_torso = 0.025
target_env_friction = 1.0
target_env_mass_torso = 0.075
target_env_choice = 'target3'
dataset_type = 'imitation'
source_seed = 0
target_seed = 100

############# REINFORCE #############
trained_dve_path = None
reinforce_layers = [256, 256]
dve_lr = 0.0001
outer_iterations = int(20e3)
inner_iterations = 5
epsilon = 1e-8
threshold = 0.9
modify_ratio = 0.9

############# CL #############
cl_hop = 5
freeze_cl_epoch = int(max_epoch / 4)
update_cl_interval = 100

############# with filtering #############
filter_out_ratio = 0.1

############# trajectory valuation #############
std_scale = 0.1
target_max_epoch = 50
temperature = 1.0
