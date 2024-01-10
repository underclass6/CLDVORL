import torch
import wandb
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

import sys
sys.path.append('./')
from offlinerllib.buffer import D4RLTransitionBuffer
from offlinerllib.module.actor import ClippedGaussianActor
from offlinerllib.module.critic import Critic, DoubleCritic
from offlinerllib.module.net.mlp import MLP
from offlinerllib.policy.model_free import XQLPolicy
from offlinerllib.utils.d4rl import get_d4rl_dataset
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


def vanilla(args):
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    offline_buffer = D4RLTransitionBuffer(dataset)

    actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
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
        norm_layer=args.norm_layer,
        dropout=args.value_dropout
    ).to(args.device)

    policy = XQLPolicy(
        actor=actor, critic_q=critic_q, critic_v=critic_v,
        num_v_update=args.num_v_update,
        scale_random_sample=args.scale_random_sample,
        loss_temperature=args.loss_temperature,
        aw_temperature=args.aw_temperature,
        use_log_loss=args.use_log_loss,
        noise_std=args.noise_std,
        tau=args.tau,
        discount=args.discount,
        max_action=args.max_action,
        max_clip=args.max_clip,
        device=args.device
    ).to(args.device)
    actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
    policy.configure_optimizers(
        actor_lr=args.actor_lr,
        critic_v_lr=args.critic_v_lr,
        critic_q_lr=args.critic_q_lr,
        actor_opt_scheduler_steps=actor_opt_scheduler_steps
    )

    ### Log
    exp_name = "_".join([args.d4rl_source_env, args.d4rl_target_env, 'vanilla', "seed"+str(args.seed)])
    logger = CompositeLogger(log_path=f"./results/xql/vanilla", name=exp_name, loggers_config={
        "FileLogger": {"activate": not args.debug},
        "TensorboardLogger": {"activate": not args.debug},
        "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    })

    ### Prepare source & target dataset
    source_env, source_buffer_dataset = get_d4rl_dataset(args.d4rl_source_env, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    env, target_buffer_dataset = get_d4rl_dataset(args.d4rl_target_env, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    source_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(source_buffer_dataset['observations']))):
        source_buffer_o.add(
            source_buffer_dataset['observations'][i],
            source_buffer_dataset['actions'][i],
            source_buffer_dataset['next_observations'][i],
            source_buffer_dataset['rewards'][i],
            source_buffer_dataset['terminals'][i]
        )
    source_buffer_o.create_trajs()
    target_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(target_buffer_dataset['observations']))):
        target_buffer_o.add(
            target_buffer_dataset['observations'][i],
            target_buffer_dataset['actions'][i],
            target_buffer_dataset['next_observations'][i],
            target_buffer_dataset['rewards'][i],
            target_buffer_dataset['terminals'][i]
        )
    target_buffer_o.create_trajs()
    # construct the combined source replay buffer
    source_buffer, target_buffer = get_combined_buffer(env, source_buffer_o, target_buffer_o, ratio=args.split_ratio, device=args.dev)
    train_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    train_buffer.merge_buffers(source_buffer, target_buffer)

    _, mixed_dataset, _, _ = get_mixed_d4rl_mujoco_datasets_from(args.d4rl_source_env.split('-')[0],
                                                              args.d4rl_source_env.split('-')[1],
                                                              args.d4rl_target_env.split('-')[1], 1e6, args.split_ratio, source_env,
                                                              source_buffer_dataset, keep_traj=True,
                                                              normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)

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
                              path=f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/seed{args.seed}/policy")

    np.save(logger.log_path + f'/avg-rewards-seed{args.seed}.npy', avg_rewards)

    wandb.finish()


def without_CL(args):
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    offline_buffer = D4RLTransitionBuffer(dataset)

    actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
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
        norm_layer=args.norm_layer,
        dropout=args.value_dropout
    ).to(args.device)

    policy = XQLPolicy(
        actor=actor, critic_q=critic_q, critic_v=critic_v,
        num_v_update=args.num_v_update,
        scale_random_sample=args.scale_random_sample,
        loss_temperature=args.loss_temperature,
        aw_temperature=args.aw_temperature,
        use_log_loss=args.use_log_loss,
        noise_std=args.noise_std,
        tau=args.tau,
        discount=args.discount,
        max_action=args.max_action,
        max_clip=args.max_clip,
        device=args.device
    ).to(args.device)
    actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
    policy.configure_optimizers(
        actor_lr=args.actor_lr,
        critic_v_lr=args.critic_v_lr,
        critic_q_lr=args.critic_q_lr,
        actor_opt_scheduler_steps=actor_opt_scheduler_steps
    )

    ### Log
    exp_name = "_".join([args.d4rl_source_env, args.d4rl_target_env, 'withoutCL', "seed"+str(args.seed)])
    logger = CompositeLogger(log_path=f"./results/xql/withoutCL", name=exp_name, loggers_config={
        "FileLogger": {"activate": not args.debug},
        "TensorboardLogger": {"activate": not args.debug},
        "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
    })

    ### Prepare source & target dataset
    source_env, source_buffer_dataset = get_d4rl_dataset(args.d4rl_source_env, normalize_obs=args.normalize_obs,
                                                         normalize_reward=args.normalize_reward)
    env, target_buffer_dataset = get_d4rl_dataset(args.d4rl_target_env, normalize_obs=args.normalize_obs,
                                                  normalize_reward=args.normalize_reward)
    source_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(source_buffer_dataset['observations']))):
        source_buffer_o.add(
            source_buffer_dataset['observations'][i],
            source_buffer_dataset['actions'][i],
            source_buffer_dataset['next_observations'][i],
            source_buffer_dataset['rewards'][i],
            source_buffer_dataset['terminals'][i]
        )
    source_buffer_o.create_trajs()
    target_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(target_buffer_dataset['observations']))):
        target_buffer_o.add(
            target_buffer_dataset['observations'][i],
            target_buffer_dataset['actions'][i],
            target_buffer_dataset['next_observations'][i],
            target_buffer_dataset['rewards'][i],
            target_buffer_dataset['terminals'][i]
        )
    target_buffer_o.create_trajs()
    # construct the combined source replay buffer
    source_buffer, target_buffer = get_combined_buffer(env, source_buffer_o, target_buffer_o, ratio=args.split_ratio,
                                                       device=args.dev)

    ###  Train Delta Classifier
    delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
    delta_model_path = f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}"
    # Define the model file names
    cf_model_files = ['cla_sa', 'cla_sas']
    # Check if the model files exist
    if all(Path(delta_model_path + '/delta_models/' + f).is_file() for f in cf_model_files):
        # If the model files exist, load them
        print('Loading delta classifiers...')
        delta.load_delta_models(delta_model_path)
        print('Finished loading delta classifiers')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training delta classifiers...')
        delta.train(source_buffer, target_buffer, args)
        # for model_file in cf_model_files:
        delta.save_delta_models(delta_model_path)
        print('Finished training delta classifiers')

    ### select trajectories and order them according to values
    target_buffer_subset = get_subset_target_buffer(env=env, target_buffer=target_buffer, device=args.dev)

    results_dir = f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/dve"
    dvrl = DVRL(source_buffer, target_buffer_subset, args.dev, env, results_dir, args.ex_configs, args)
    # Define the model file names
    dve_model_files = ['reinforce_final', 'reinforce_optimizer_final']
    # Check if the model files exist
    if all(Path("%s/dvrl_models/Trained_With_Seed_%d_Friction_%f_Mass_%f_Gamma_%f/" % \
                (results_dir, args.source_seed, args.source_env_friction,
                 args.source_env_mass_torso, args.discount) + f).is_file() for f in dve_model_files):
        # If the model files exist, load them
        print('Loading DVE...')
        dvrl.load_dve(dvrl.model_dir, type='final')
    else:
        # If the model files do not exist, train the models and save them
        print('Start training DVE...')
        dvrl.train()
        print('Finished training DVE...')

    ## use only source buffer
    source_buffer = source_buffer_o
    target_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)

    dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)
    source_buffer = modify_rewards(source_buffer, dve_out, ratio=args.modify_ratio)

    train_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    train_buffer.merge_buffers(source_buffer, target_buffer)

    for i in range(len(source_buffer_dataset['observations'])):
        source_buffer_dataset['rewards'][i] = source_buffer.reward[i]

    _, mixed_dataset, _, _ = get_mixed_d4rl_mujoco_datasets_from(args.d4rl_source_env.split('-')[0],
                                                              args.d4rl_source_env.split('-')[1],
                                                              args.d4rl_target_env.split('-')[1], 1e6, args.split_ratio, source_env,
                                                              source_buffer_dataset, keep_traj=True,
                                                              normalize_obs=False, normalize_reward=False)

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
                              path=f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/seed{args.seed}/policy")

    np.save(logger.log_path + f'/avg-rewards-seed{args.seed}.npy', avg_rewards)

    wandb.finish()


def traj_valuation(args):
    env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[-1]

    ### Log
    if args.run_dcla_ratio:
        logger = CompositeLogger(
            log_path=f"./ablation_results/xql/{args.d4rl_source_env.split('-')[0]}/{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}/dcla_ratio/{args.dcla_ratio}",
            name=f"xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_dcla_ratio_{args.dcla_ratio}_seed_{args.seed}",
            loggers_config={
                "FileLogger": {"activate": not args.debug},
                "TensorboardLogger": {"activate": not args.debug},
                "WandbLogger": {"activate": not args.debug, "config": args,
                                "settings": wandb.Settings(_disable_stats=True), **args.wandb}
            })
    elif args.run_modify_ratio:
        logger = CompositeLogger(
            log_path=f"./ablation_results/xql/{args.d4rl_source_env.split('-')[0]}/{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}/modify_ratio/{args.modify_ratio}/",
            name=f"xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_modify_ratio_{args.modify_ratio}_seed_{args.seed}",
            loggers_config={
                "FileLogger": {"activate": not args.debug},
                "TensorboardLogger": {"activate": not args.debug},
                "WandbLogger": {"activate": not args.debug, "config": args,
                                "settings": wandb.Settings(_disable_stats=True), **args.wandb}
            })
    elif args.run_std_scale:
        logger = CompositeLogger(
            log_path=f"./ablation_results/xql/{args.d4rl_source_env.split('-')[0]}/{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}/std_scale/{args.std_scale}/",
            name=f"xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_std_scale_{args.std_scale}_seed_{args.seed}",
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
    source_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(source_buffer_dataset['observations']))):
        source_buffer_o.add(
            source_buffer_dataset['observations'][i],
            source_buffer_dataset['actions'][i],
            source_buffer_dataset['next_observations'][i],
            source_buffer_dataset['rewards'][i],
            source_buffer_dataset['terminals'][i]
        )
    source_buffer_o.create_trajs()
    target_buffer_o = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in tqdm(range(len(target_buffer_dataset['observations']))):
        target_buffer_o.add(
            target_buffer_dataset['observations'][i],
            target_buffer_dataset['actions'][i],
            target_buffer_dataset['next_observations'][i],
            target_buffer_dataset['rewards'][i],
            target_buffer_dataset['terminals'][i]
        )
    target_buffer_o.create_trajs()
    # construct the combined source replay buffer
    source_buffer, target_buffer = get_combined_buffer(env, source_buffer_o, target_buffer_o, ratio=args.split_ratio,
                                                       device=args.dev)
    train_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    train_buffer.merge_buffers(source_buffer, target_buffer)

    _, mixed_dataset, source_dataset, target_dataset = get_mixed_d4rl_mujoco_datasets_from(args.d4rl_source_env.split('-')[0],
                                                                 args.d4rl_source_env.split('-')[1],
                                                                 args.d4rl_target_env.split('-')[1], 1e6, args.split_ratio,
                                                                 source_env,
                                                                 source_buffer_dataset, keep_traj=True,
                                                                 normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)

    # -----------------------------------------------------------------------------------------------
    ###  Train Delta Classifier
    delta = DeltaCla(env.observation_space.shape[0], env.action_space.shape[0], args)
    delta_model_path = f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}"
    # Define the model file names
    cf_model_files = ['cla_sa', 'cla_sas']
    # If the model files do not exist, train the models and save them
    print('Start training delta classifiers...')
    delta.train(source_buffer, target_buffer, args)
    print('Finished training delta classifiers')

    ### select trajectories and order them according to values
    target_buffer_subset = get_subset_target_buffer(env=env, target_buffer=target_buffer, device=args.dev)

    results_dir = f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/dve"
    dvrl = DVRL(source_buffer, target_buffer_subset, args.dev, env, results_dir, args.ex_configs, args)
    # Define the model file names
    dve_model_files = ['reinforce_final', 'reinforce_optimizer_final']
    # If the model files do not exist, train the models and save them
    print('Start training DVE...')
    dvrl.train()
    print('Finished training DVE...')

    # modify reward
    source_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.dev)
    for i in range(source_dataset['rewards'].shape[0]):
        source_buffer.add(
            source_dataset['observations'][i],
            source_dataset['actions'][i],
            source_dataset['next_observations'][i],
            source_dataset['rewards'][i],
            source_dataset['terminals'][i]
        )
    dve_out, sel_vec = dvrl.data_valuate(source_buffer, args.batch_size)

    for i in range(source_dataset['rewards'].shape[0]):
        source_dataset['rewards'][i] = (1 - args.modify_ratio) * source_dataset['rewards'][i] + args.modify_ratio * \
                                       dve_out[i]
    # -----------------------------------------------------------------------------------------------


    ### train agent of target domain over target offline dataset
    actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
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
        norm_layer=args.norm_layer,
        dropout=args.value_dropout
    ).to(args.device)

    policy_target = XQLPolicy(
        actor=actor, critic_q=critic_q, critic_v=critic_v,
        num_v_update=args.num_v_update,
        scale_random_sample=args.scale_random_sample,
        loss_temperature=args.loss_temperature,
        aw_temperature=args.aw_temperature,
        use_log_loss=args.use_log_loss,
        noise_std=args.noise_std,
        tau=args.tau,
        discount=args.discount,
        max_action=args.max_action,
        max_clip=args.max_clip,
        device=args.device
    ).to(args.device)
    actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
    policy_target.configure_optimizers(
        actor_lr=args.actor_lr,
        critic_v_lr=args.critic_v_lr,
        critic_q_lr=args.critic_q_lr,
        actor_opt_scheduler_steps=actor_opt_scheduler_steps
    )

    offline_buffer = D4RLTransitionBuffer(target_dataset)

    avg_rewards = []  # store eval_metrics

    policy_target.train()
    for i_epoch in trange(1, args.target_max_epoch + 1):
        for i_step in range(args.step_per_epoch):
            batch = offline_buffer.random_batch(args.batch_size)
            train_metrics = policy_target.update(batch)

        if i_epoch % args.eval_interval == 0:
            eval_metrics = eval_offline_policy(env, policy_target, args.eval_episode, seed=args.seed)

            print(f"Episode {i_epoch}: \n{eval_metrics}")

    ### valaute trajectory
    def valuate_traj(traj, policy_source, policy_target):

        ### option 1 ###
        # obs = torch.from_numpy(traj['observations']).to('cuda')
        # acts = torch.from_numpy(traj['actions']).to('cuda')
        # next_obs = torch.from_numpy(traj['next_observations']).to('cuda')
        # rewards = torch.from_numpy(traj['rewards']).to('cuda')
        #
        # std_scale = 0.1
        # act_mean_source, *_ = policy_source.actor.sample(obs, True)
        # dist_source = torch.distributions.normal.Normal(act_mean_source, std_scale * torch.ones_like(act_mean_source))
        # log_prob_source = dist_source.log_prob(acts).sum(-1, keepdim=True)
        # act_mean_target, *_ = policy_target.actor.sample(obs, True)
        # dist_target = torch.distributions.normal.Normal(act_mean_target, std_scale * torch.ones_like(act_mean_target))
        # log_prob_target = dist_target.log_prob(acts).sum(-1, keepdim=True)
        #
        # log_prob_source = torch.log(torch.exp(log_prob_source) / torch.sum(torch.exp(log_prob_source)))  # normalize sum to 1
        # log_prob_target = torch.log(torch.exp(log_prob_target) / torch.sum(torch.exp(log_prob_target)))
        #
        # # log_prob_source = policy_source.actor.evaluate(obs, acts)
        # # log_prob_target = policy_target.actor.evaluate(obs, acts)
        #
        # kld = torch.nn.functional.kl_div(torch.squeeze(log_prob_source), torch.squeeze(log_prob_target),
        #                                  log_target=True, reduction='mean')
        #
        # value = np.exp(-kld.item())
        ### option 1 ###

        ### option 2 ###
        obs = torch.from_numpy(traj['observations']).to('cuda')
        acts = torch.from_numpy(traj['actions']).to('cuda')
        next_obs = torch.from_numpy(traj['next_observations']).to('cuda')
        rewards = torch.from_numpy(traj['rewards']).to('cuda')

        std_scale = args.std_scale
        act_mean_source, *_ = policy_source.actor.sample(obs, True)
        dist_source = torch.distributions.normal.Normal(act_mean_source, std_scale * torch.ones_like(act_mean_source))
        log_prob_source = dist_source.log_prob(acts).sum(-1, keepdim=True)
        act_mean_target, *_ = policy_target.actor.sample(obs, True)
        dist_target = torch.distributions.normal.Normal(act_mean_target, std_scale * torch.ones_like(act_mean_target))
        log_prob_target = dist_target.log_prob(acts).sum(-1, keepdim=True)

        log_prob_source = torch.log(
            torch.exp(log_prob_source) / torch.sum(torch.exp(log_prob_source)))  # normalize sum to 1
        log_prob_target = torch.log(torch.exp(log_prob_target) / torch.sum(torch.exp(log_prob_target)))

        # log_prob_source = policy_source.actor.evaluate(obs, acts)
        # log_prob_target = policy_target.actor.evaluate(obs, acts)

        kld = torch.nn.functional.kl_div(torch.squeeze(log_prob_source), torch.squeeze(log_prob_target),
                                         log_target=True, reduction='mean')

        value = np.exp(-kld.item()) * np.sum(traj['rewards'])
        ### option 2 ###

        return value

    ### train agent of source domain
    actor_backend = MLP(input_dim=obs_shape, hidden_dims=args.hidden_dims, dropout=args.dropout)
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
        norm_layer=args.norm_layer,
        dropout=args.value_dropout
    ).to(args.device)

    policy_source = XQLPolicy(
        actor=actor, critic_q=critic_q, critic_v=critic_v,
        num_v_update=args.num_v_update,
        scale_random_sample=args.scale_random_sample,
        loss_temperature=args.loss_temperature,
        aw_temperature=args.aw_temperature,
        use_log_loss=args.use_log_loss,
        noise_std=args.noise_std,
        tau=args.tau,
        discount=args.discount,
        max_action=args.max_action,
        max_clip=args.max_clip,
        device=args.device
    ).to(args.device)
    actor_opt_scheduler_steps = args.max_epoch * args.step_per_epoch if args.actor_opt_decay_schedule == "cosine" else None
    policy_source.configure_optimizers(
        actor_lr=args.actor_lr,
        critic_v_lr=args.critic_v_lr,
        critic_q_lr=args.critic_q_lr,
        actor_opt_scheduler_steps=actor_opt_scheduler_steps
    )

    # get trajectories from source offline dataset
    begin_idx = 0
    source_trajs = []
    for idx, te in enumerate(source_dataset['terminals']):
        if te:
            end_idx = idx
            trajs = dict()
            trajs['observations'] = source_dataset['observations'][begin_idx:end_idx+1]
            trajs['actions'] = source_dataset['actions'][begin_idx:end_idx + 1]
            trajs['next_observations'] = source_dataset['next_observations'][begin_idx:end_idx + 1]
            trajs['rewards'] = source_dataset['rewards'][begin_idx:end_idx + 1]
            trajs['terminals'] = source_dataset['terminals'][begin_idx:end_idx + 1]
            source_trajs.append(trajs)
            begin_idx = end_idx + 1
    source_trajs_len = len(source_trajs)

    if args.d4rl_source_env.split('-')[0] == 'halfcheetah':
        begin_idx = 0
        source_trajs = []
        hop = int(source_dataset['terminals'].shape[0] / 500)
        while begin_idx < source_dataset['terminals'].shape[0]:
            end_idx = begin_idx + hop
            trajs = dict()
            trajs['observations'] = source_dataset['observations'][begin_idx:end_idx + 1]
            trajs['actions'] = source_dataset['actions'][begin_idx:end_idx + 1]
            trajs['next_observations'] = source_dataset['next_observations'][begin_idx:end_idx + 1]
            trajs['rewards'] = source_dataset['rewards'][begin_idx:end_idx + 1]
            trajs['terminals'] = source_dataset['terminals'][begin_idx:end_idx + 1]
            source_trajs.append(trajs)
            begin_idx = end_idx + 1
        source_trajs_len = len(source_trajs)

    # get trajectories from target offline dataset
    ends = np.arange(0, target_dataset['terminals'].shape[0], int(len(source_trajs) * 0.2))[1:]
    for idx in ends:
        target_dataset['terminals'][idx] = True

    begin_idx = 0
    target_trajs = []
    for idx, te in enumerate(target_dataset['terminals']):
        if te:
            end_idx = idx
            trajs = dict()
            trajs['observations'] = target_dataset['observations'][begin_idx:end_idx + 1]
            trajs['actions'] = target_dataset['actions'][begin_idx:end_idx + 1]
            trajs['next_observations'] = target_dataset['next_observations'][begin_idx:end_idx + 1]
            trajs['rewards'] = target_dataset['rewards'][begin_idx:end_idx + 1]
            trajs['terminals'] = target_dataset['terminals'][begin_idx:end_idx + 1]
            target_trajs.append(trajs)
            begin_idx = end_idx + 1
    target_trajs_len = len(target_trajs)

    for idx in ends:
        target_dataset['terminals'][idx] = False

    source_trajs.extend(target_trajs)

    temperature = args.temperature
    num_iter = 0
    num_trajs = int((source_trajs_len + target_trajs_len) / 5)
    e_size = 10
    for e in range(e_size):
        # compute the value of each trajectory in source offline dataset
        trajs_value = []
        for traj in source_trajs:
            v = valuate_traj(traj, policy_source, policy_target)
            trajs_value.append(v)

        # train using sampled trajectories
        trajs_value_scaled = torch.Tensor(trajs_value) / temperature

        sorted_value, value_ids = torch.sort(trajs_value_scaled, descending=True)

        iter_num = 50
        for i in range(0, len(value_ids), num_trajs):
            trajs_ids = value_ids[i:i+num_trajs]

            source_dataset_sampled = dict()
            for idx in trajs_ids:
                traj = source_trajs[idx]
                if 'observations' not in source_dataset_sampled:
                    source_dataset_sampled['observations'] = traj['observations']
                    source_dataset_sampled['actions'] = traj['actions']
                    source_dataset_sampled['next_observations'] = traj['next_observations']
                    source_dataset_sampled['rewards'] = traj['rewards']
                    source_dataset_sampled['terminals'] = traj['terminals']
                else:
                    source_dataset_sampled['observations'] = np.append(source_dataset_sampled['observations'],
                                                                       traj['observations'], axis=0)
                    source_dataset_sampled['actions'] = np.append(source_dataset_sampled['actions'], traj['actions'],
                                                                  axis=0)
                    source_dataset_sampled['next_observations'] = np.append(source_dataset_sampled['next_observations'],
                                                                            traj['next_observations'], axis=0)
                    source_dataset_sampled['rewards'] = np.append(source_dataset_sampled['rewards'], traj['rewards'],
                                                                  axis=0)
                    source_dataset_sampled['terminals'] = np.append(source_dataset_sampled['terminals'],
                                                                    traj['terminals'], axis=0)
            ### add target offline dataset ###
            source_dataset_sampled['observations'] = np.append(source_dataset_sampled['observations'],
                                                               target_dataset['observations'], axis=0)
            source_dataset_sampled['actions'] = np.append(source_dataset_sampled['actions'], target_dataset['actions'],
                                                          axis=0)
            source_dataset_sampled['next_observations'] = np.append(source_dataset_sampled['next_observations'],
                                                                    target_dataset['next_observations'], axis=0)
            source_dataset_sampled['rewards'] = np.append(source_dataset_sampled['rewards'], target_dataset['rewards'],
                                                          axis=0)
            source_dataset_sampled['terminals'] = np.append(source_dataset_sampled['terminals'],
                                                            target_dataset['terminals'], axis=0)
            ### add target offline dataset ###
            offline_buffer = D4RLTransitionBuffer(source_dataset_sampled)

            iter_num -= 10
            policy_source.train()
            for i_epoch in trange(1, iter_num + 1):
                for i_step in range(args.step_per_epoch):
                    batch = offline_buffer.random_batch(args.batch_size)
                    train_metrics = policy_source.update(batch)

                if i_epoch % args.eval_interval == 0:
                    eval_metrics = eval_offline_policy(env, policy_source, args.eval_episode, seed=args.seed)

                    logger.info(f"Episode {num_iter}: \n{eval_metrics}")

                if i_epoch % args.log_interval == 0:
                    logger.log_scalars("", train_metrics, step=num_iter)
                    logger.log_scalars("Eval", eval_metrics, step=num_iter)

                    avg_rewards.append(eval_metrics)

                if i_epoch % args.save_interval == 0:
                    logger.log_object(name=f"policy_{num_iter}.pt", object=policy_source.state_dict(),
                                      path=f"./models/xql/{args.task}/{args.d4rl_source_env}_{args.d4rl_target_env}/seed{args.seed}/policy")

                num_iter += 1

    if args.run_dcla_ratio:
        np.save(
            logger.log_path + f"/avg_rewards_xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_dcla_ratio_{args.dcla_ratio}_seed_{args.seed}.npy",
            avg_rewards)
    elif args.run_modify_ratio:
        np.save(
            logger.log_path + f"/avg_rewards_xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_modify_ratio_{args.modify_ratio}_seed_{args.seed}.npy",
            avg_rewards)
    elif args.run_std_scale:
        np.save(
            logger.log_path + f"/avg_rewards_xql_{args.d4rl_source_env.split('-')[0]}_{args.d4rl_source_env.split('-')[1]}_{args.d4rl_target_env.split('-')[1]}_std_scale_{args.std_scale}_seed_{args.seed}.npy",
            avg_rewards)

    wandb.finish()


def main(pass_in=None):
    args = parse_args('./reproduce/xql/config/cldv/base.py')
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    source_datasets = [
        'hopper-random-v2',
        'walker2d-random-v2',
        'halfcheetah-random-v2'
    ]
    target_datasets = [
        'hopper-expert-v2',
        'walker2d-expert-v2',
        'halfcheetah-expert-v2'
    ]

    arguments = {}
    for sd, td in zip(source_datasets, target_datasets):
        arguments[(sd, td)] = parse_args('./reproduce/xql/config/cldv/base.py')

    # hopper
    # hopper-random-v2, hopper-expert-v2
    args = arguments[('hopper-random-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # walker2d
    # walker2d-random-v2, walker2d-expert-v2
    args = arguments[('walker2d-random-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # halfcheetah
    # halfcheetah-random-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    if hasattr(args, 'dataset'):  # e.g. --dataset hopper-random-v2_hopper-medium-v2,walker2d-random-v2_walker2d-medium-v2
        source_datasets = []
        target_datasets = []
        st_pairs = args.dataset.split(',')
        dataset = [data.split('_') for data in st_pairs]
        for sd, td in dataset:
            source_datasets.append(sd)
            target_datasets.append(td)

    if hasattr(args, 'exp_type'):
        run_traj_valuation = False
        ets = args.exp_type.split('_')
        for et in ets:
            if et == 'trajValuation':
                run_traj_valuation = True
    else:
        run_traj_valuation = True

    # ablation study over parameters
    if hasattr(args, 'dcla_ratios'):
        dcla_ratios = list(args.dcla_ratios)
    else:
        dcla_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    if hasattr(args, 'modify_ratios'):
        modify_ratios = list(args.modify_ratios)
    else:
        modify_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    if hasattr(args, 'std_scales'):
        std_scales = list(args.std_scales)
    else:
        std_scales = [0.01, 0.1, 1, 10, 100]

    run_exp = {
        'dcla_ratios': True,
        'modify_ratios': True,
        'std_scales': True,
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

                if run_traj_valuation:
                    # run traj_valuation
                    traj_valuation(args)

    # hopper
    # hopper-random-v2, hopper-expert-v2
    args = arguments[('hopper-random-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # walker2d
    # walker2d-random-v2, walker2d-expert-v2
    args = arguments[('walker2d-random-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # halfcheetah
    # halfcheetah-random-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    if run_exp['modify_ratios']:
        for mr in modify_ratios:
            for sd, td in zip(source_datasets, target_datasets):
                # last chance to modify args
                args = arguments[(sd, td)]
                args.alpha = 2.5
                args.normalize_obs = True
                args.normalize_reward = False
                args.task = sd
                args.d4rl_source_env = sd
                args.d4rl_target_env = td
                args.modify_ratio = mr
                args.run_modify_ratio = True

                if run_traj_valuation:
                    # run traj_valuation
                    traj_valuation(args)

    # hopper
    # hopper-random-v2, hopper-expert-v2
    args = arguments[('hopper-random-v2', 'hopper-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # walker2d
    # walker2d-random-v2, walker2d-expert-v2
    args = arguments[('walker2d-random-v2', 'walker2d-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    # halfcheetah
    # halfcheetah-random-v2, halfcheetah-expert-v2
    args = arguments[('halfcheetah-random-v2', 'halfcheetah-expert-v2')]
    args.dcla_epochs = 50
    args.dcla_hidden_size = 512
    args.dcla_ratio = 0.7
    args.modify_ratio = 0.05
    args.outer_iterations = 20000
    args.run_dcla_ratio = False
    args.run_modify_ratio = False
    args.run_std_scale = False

    if run_exp['std_scales']:
        for ss in std_scales:
            for sd, td in zip(source_datasets, target_datasets):
                # last chance to modify args
                args = arguments[(sd, td)]
                args.alpha = 2.5
                args.normalize_obs = True
                args.normalize_reward = False
                args.task = sd
                args.d4rl_source_env = sd
                args.d4rl_target_env = td
                args.std_scale = ss
                args.run_std_scale = True

                if run_traj_valuation:
                    # run traj_valuation
                    traj_valuation(args)


if __name__ == '__main__':
    main()
