from UtilsRL.misc import NameSpace

seed = 42
device = None
task = None

max_epoch = 1000
step_per_epoch = 1000
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50
max_action = 1.0

hidden_dims = [256, 256, 256]
discount = 0.99
tau = 0.005
eta = 1.0
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
auto_alpha = True
batch_size = 256
policy_logstd_min = -5

normalize_obs = False
normalize_reward = False

do_reverse_update = True

name = "corl"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

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
