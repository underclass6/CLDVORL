import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

import sys
sys.path.append('./')
from offlinerllib.buffer import D4RLTransitionBuffer
from offlinerllib.module.actor import ClippedGaussianActor, SquashedDeterministicActor
from offlinerllib.module.critic import Critic, DoubleCritic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import IQLPolicy
from offlinerllib.env.d4rl import get_d4rl_dataset
from offlinerllib.utils.eval import eval_offline_policy

from offlinerllib.env.mixed import get_mixed_d4rl_mujoco_datasets_from

import random
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from pathlib import Path
from CLDVORL.CLDV.delta_classifier import DeltaCla
from CLDVORL.CLDV.dvrl_test import DVRL
from CLDVORL.CLDV.replay_buffer import ReplayBuffer
from CLDVORL.CLDV.cldv import filter_trajectories, sort_trajectories, modify_rewards
from CLDVORL.CLDV.utils import get_gym, get_combined_buffer, get_d4rl_buffer, get_subset_target_buffer


def without_CL(args):
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    offline_buffer = D4RLTransitionBuffer(dataset)

    actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
    if args.iql_deterministic:
        actor = SquashedDeterministicActor(
            backend=actor_backend,
            input_dim=args.hidden_dims[-1],
            output_dim=action_shape,
        ).to(args.device)
    else:
        actor = ClippedGaussianActor(
            backend=actor_backend,
            input_dim=args.hidden_dims[-1],
            output_dim=action_shape,
            conditioned_logstd=args.conditioned_logstd,
            logstd_min=args.policy_logstd_min
        ).to(args.device)

    critic_q = DoubleCritic(
        backend=torch.nn.Identity(),
        input_dim=obs_shape + action_shape,
        hidden_dims=args.hidden_dims,
    ).to(args.device)

    critic_v = Critic(
        backend=torch.nn.Identity(),
        input_dim=obs_shape,
        hidden_dims=args.hidden_dims,
    ).to(args.device)

    policy = IQLPolicy(
        actor=actor, critic_q=critic_q, critic_v=critic_v,
        expectile=args.expectile, temperature=args.temperature,
        tau=args.tau,
        discount=args.discount,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
    policy.configure_optimizers(
        actor_lr=args.actor_lr,
        critic_v_lr=args.critic_v_lr,
        critic_q_lr=args.critic_q_lr,
        actor_opt_scheduler_steps=actor_opt_scheduler_steps
    )

    ### Log
    if args.run_dcla_ratio:
        logger = CompositeLogger(
            log_path=f"./ablation_results/iql/{args.d4rl_source_env.split('-')[0]}/{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}/dcla_ratio/{args.dcla_ratio}",
            name=f"iql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_dcla_ratio_{args.dcla_ratio}_seed_{args.seed}",
            loggers_config={
                "FileLogger": {"activate": not args.debug},
                "TensorboardLogger": {"activate": not args.debug},
                "WandbLogger": {"activate": not args.debug, "config": args,
                                "settings": wandb.Settings(_disable_stats=True), **args.wandb}
            })
    elif args.run_filter_ratio:
        logger = CompositeLogger(
            log_path=f"./ablation_results/iql/{args.d4rl_source_env.split('-')[0]}/{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}/filter_ratio/{args.filter_ratio}/",
            name=f"iql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_filter_ratio_{args.filter_ratio}_seed_{args.seed}",
            loggers_config={
                "FileLogger": {"activate": not args.debug},
                "TensorboardLogger": {"activate": not args.debug},
                "WandbLogger": {"activate": not args.debug, "config": args,
                                "settings": wandb.Settings(_disable_stats=True), **args.wandb}
            })

    ### Prepare source & target dataset
    source_env, source_buffer_dataset = get_d4rl_dataset(args.d4rl_source_env, normalize_obs=args.normalize_obs,
                                                         normalize_reward=args.normalize_reward)
    env, target_buffer_dataset = get_d4rl_dataset(args.d4rl_target_env, normalize_obs=args.normalize_obs,
                                                  normalize_reward=args.normalize_reward)
    _, _, dataset_source, dataset_target = get_mixed_d4rl_mujoco_datasets_from(args.d4rl_source_env.split('-')[0],
                                                                               args.d4rl_source_env.split('-')[1],
                                                                               args.d4rl_target_env.split('-')[1], 1e6,
                                                                               args.split_ratio, source_env,
                                                                               source_buffer_dataset, keep_traj=True,
                                                                               normalize_obs=args.normalize_obs,
                                                                               normalize_reward=args.normalize_reward)

    source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(dataset_source['observations']))):
        source_buffer.add(
            dataset_source['observations'][i],
            dataset_source['actions'][i],
            dataset_source['next_observations'][i],
            dataset_source['rewards'][i],
            dataset_source['terminals'][i]
        )
    source_buffer.create_trajs()
    target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(dataset_target['observations']))):
        target_buffer.add(
            dataset_target['observations'][i],
            dataset_target['actions'][i],
            dataset_target['next_observations'][i],
            dataset_target['rewards'][i],
            dataset_target['terminals'][i]
        )
    target_buffer.create_trajs()

    ###  Train Delta Classifier
    delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
    delta_model_path = f"./models/awac/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}"
    # Define the model file names
    cf_model_files = ['cla_sa', 'cla_sas']
    print('Start training delta classifiers...')
    delta.train(source_buffer, target_buffer, args)
    print('Finished training delta classifiers')

    ### select trajectories and order them according to values
    target_buffer_subset = get_subset_target_buffer(env=env, target_buffer=target_buffer, device=args.dev)

    results_dir = f"./models/awac/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/dve"
    dvrl = DVRL(source_buffer, target_buffer_subset, args.dev, env, results_dir, args.ex_configs, args)
    # Define the model file names
    dve_model_files = ['reinforce_final', 'reinforce_optimizer_final']
    print('Start training DVE...')
    dvrl.train()
    print('Finished training DVE...')

    dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)

    ### filter out transitions
    source_filtered_dataset = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        "ends": []
    }

    sorted_dve_out = sorted(range(len(dve_out)), key=lambda k: dve_out[k])
    start_pos = int(args.filter_ratio * len(sorted_dve_out))
    sorted_dve_out = sorted_dve_out[start_pos:]
    for idx in sorted_dve_out:
        source_filtered_dataset['observations'].append(source_buffer_dataset['observations'][idx])
        source_filtered_dataset['actions'].append(source_buffer_dataset['actions'][idx])
        source_filtered_dataset['next_observations'].append(source_buffer_dataset['next_observations'][idx])
        source_filtered_dataset['rewards'].append(source_buffer_dataset['rewards'][idx])
        source_filtered_dataset['terminals'].append(source_buffer_dataset['terminals'][idx])
        source_filtered_dataset['ends'].append(source_buffer_dataset['ends'][idx])

    mixed_dataset = source_filtered_dataset
    for i in range(len(dataset_target['observations'])):
        mixed_dataset['observations'].append(dataset_target['observations'][i])
        mixed_dataset['actions'].append(dataset_target['actions'][i])
        mixed_dataset['next_observations'].append(dataset_target['next_observations'][i])
        mixed_dataset['rewards'].append(dataset_target['rewards'][i])
        mixed_dataset['terminals'].append(dataset_target['terminals'][i])
        mixed_dataset['ends'].append(dataset_target['ends'][i])
    mixed_dataset['observations'] = np.array(mixed_dataset['observations'])
    mixed_dataset['actions'] = np.array(mixed_dataset['actions'])
    mixed_dataset['next_observations'] = np.array(mixed_dataset['next_observations'])
    mixed_dataset['rewards'] = np.array(mixed_dataset['rewards'])
    mixed_dataset['terminals'] = np.array(mixed_dataset['terminals'])
    mixed_dataset['ends'] = np.array(mixed_dataset['ends'])

    offline_buffer = D4RLTransitionBuffer(mixed_dataset)

    ### Train and Evaluate
    avg_rewards = []  # store eval_metrics

    # main loop
    policy.train()
    for i_epoch in trange(1, args.max_epoch + 1):
        for i_step in range(args.step_per_epoch):
            batch = offline_buffer.random_batch(args.batch_size)
            train_metrics = policy.update(batch)

        if i_epoch % args.eval_interval == 0:
            eval_metrics = eval_offline_policy(env, policy, args.eval_episode, seed=args.seed)

            logger.info(f"Episode {i_epoch}: \n{eval_metrics}")

        if i_epoch % args.log_interval == 0:
            logger.log_scalars("", train_metrics, step=i_epoch)
            logger.log_scalars("Eval", eval_metrics, step=i_epoch)

            avg_rewards.append(eval_metrics)

        if i_epoch % args.save_interval == 0:
            logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(),
                              path=f"./models/iql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/seed{args.seed}/policy")

    np.save(logger.log_path + f'/avg-rewards-seed{args.seed}.npy', avg_rewards)

    wandb.finish()


def main(pass_in=None):
    args = parse_args('./reproduce/iql/config/cldv/base.py')
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    source_datasets = [
        'hopper-random-v2',
        'hopper-random-v2',
        'hopper-medium-v2',
        'walker2d-random-v2',
        'walker2d-random-v2',
        'walker2d-medium-v2',
        'halfcheetah-random-v2',
        'halfcheetah-random-v2',
        'halfcheetah-medium-v2'
    ]
    target_datasets = [
        'hopper-medium-v2',
        'hopper-expert-v2',
        'hopper-expert-v2',
        'walker2d-medium-v2',
        'walker2d-expert-v2',
        'walker2d-expert-v2',
        'halfcheetah-medium-v2',
        'halfcheetah-expert-v2',
        'halfcheetah-expert-v2'
    ]

    arguments = {}
    for sd, td in zip(source_datasets, target_datasets):
        arguments[(sd, td)] = parse_args('./reproduce/iql/config/cldv/base.py')

    # hopper
    # hopper-random-v2, hopper-medium-v2
    args = arguments[('hopper-random-v2', 'hopper-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # hopper-random-v2, hopper-expert-v2
    args = arguments[('hopper-random-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # hopper-medium-v2, hopper-expert-v2
    args = arguments[('hopper-medium-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d
    # walker2d-random-v2, walker2d-medium-v2
    args = arguments[('walker2d-random-v2', 'walker2d-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d-random-v2, walker2d-expert-v2
    args = arguments[('walker2d-random-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d-medium-v2, walker2d-expert-v2
    args = arguments[('walker2d-medium-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah
    # halfcheetah-random-v2, halfcheetah-medium-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah-random-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah-medium-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-medium-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    if hasattr(args, 'dataset'):  # e.g. --dataset hopper-random-v2_hopper-medium-v2,walker2d-random-v2_walker2d-medium-v2
        source_datasets = []
        target_datasets = []
        st_pairs = args.dataset.split(',')
        dataset = [data.split('_') for data in st_pairs]
        for sd, td in dataset:
            source_datasets.append(sd)
            target_datasets.append(td)

    if hasattr(args, 'exp_type'):
        run_without_CL = False
        ets = args.exp_type.split('_')
        for et in ets:
            if et == 'withoutCL':
                run_without_CL = True
    else:
        run_without_CL = True

    # ablation study over parameters
    if hasattr(args, 'dcla_ratios'):
        dcla_ratios = list(args.dcla_ratios)
    else:
        dcla_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    if hasattr(args, 'filter_ratios'):
        filter_ratios = list(args.filter_ratios)
    else:
        filter_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    run_exp = {
        'dcla_ratios': True,
        'filter_ratios': True,
    }
    if hasattr(args, 'ablation_types'):  # separated by comma
        for re in run_exp:
            run_exp[re] = False
        a_types = args.ablation_types.split(',')
        for at in a_types:
            run_exp[at] = True

    if run_exp['dcla_ratios']:
        for dr in dcla_ratios:
            for sd, td in zip(source_datasets, target_datasets):
                # last chance to modify args
                args = arguments[(sd, td)]
                args.alpha = 2.5
                args.normalize_obs = True
                args.normalize_reward = False
                args.task = sd
                args.d4rl_source_env = sd
                args.d4rl_target_env = td
                args.dcla_ratio = dr
                args.run_dcla_ratio = True

                print(f'dcla_ratio: {args.dcla_ratio}')
                print(f'filter_ratio: {args.filter_ratio}')

                if run_without_CL:
                    # run without_CL
                    without_CL(args)

    # hopper
    # hopper-random-v2, hopper-medium-v2
    args = arguments[('hopper-random-v2', 'hopper-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # hopper-random-v2, hopper-expert-v2
    args = arguments[('hopper-random-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # hopper-medium-v2, hopper-expert-v2
    args = arguments[('hopper-medium-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d
    # walker2d-random-v2, walker2d-medium-v2
    args = arguments[('walker2d-random-v2', 'walker2d-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d-random-v2, walker2d-expert-v2
    args = arguments[('walker2d-random-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # walker2d-medium-v2, walker2d-expert-v2
    args = arguments[('walker2d-medium-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah
    # halfcheetah-random-v2, halfcheetah-medium-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-medium-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah-random-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    # halfcheetah-medium-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-medium-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.filter_ratio = 0.0
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_filter_ratio = False

    if run_exp['filter_ratios']:
        for fr in filter_ratios:
            for sd, td in zip(source_datasets, target_datasets):
                # last chance to modify args
                args = arguments[(sd, td)]
                args.alpha = 2.5
                args.normalize_obs = True
                args.normalize_reward = False
                args.task = sd
                args.d4rl_source_env = sd
                args.d4rl_target_env = td
                args.filter_ratio = fr
                args.run_filter_ratio = True

                print(f'dcla_ratio: {args.dcla_ratio}')
                print(f'filter_ratio: {args.filter_ratio}')

                if run_without_CL:
                    # run without_CL
                    without_CL(args)


if __name__ == '__main__':
    main()


