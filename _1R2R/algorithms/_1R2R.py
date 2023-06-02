import os
import math
import pickle
from collections import OrderedDict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb
import sys
import copy

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from _1R2R.models.constructor import construct_model, format_samples_for_training
from _1R2R.models.fake_env import FakeEnv
from _1R2R.utils.logging import Progress
from _1R2R.utils.risk import wang_cdf
import _1R2R.utils.utils as utl
import _1R2R.utils.filesystem as filesystem
import _1R2R.off_policy.loader as loader


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class _1R2R(RLAlgorithm):
    """
    Author: Marc Rigter, 2022, mrigter@robots.ox.ac.uk

    This implementation builds upon the implementation of MOPO
    (https://github.com/tianheyu927/mopo) which is itself adapted from the
    implementation of soft actor-critic at
    https://github.com/rail-berkeley/softlearning.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,
            dataset_dir=None,

            normalize_states=True,
            normalize_rewards=False,
            dynamic_risk='neutral',
            rollout_repeats=10,
            num_rollout_batches=1,

            critic_lr=3e-4,
            actor_lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            pretrain_bc=False,
            bc_lr=1e-4,
            bc_epochs=50,

            deterministic=False,
            rollout_random=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            rollout_length=1,
            hidden_dim=200,
            max_model_t=None,
            model_type='mlp',
            separate_mean_var=False,
            identity_terminal=0,

            pool_load_path='',
            model_name=None,
            model_load_dir=None,
            **kwargs,
    ):
        super(_1R2R, self).__init__(**kwargs)
        self._obs_dim = np.prod(training_environment.active_observation_shape)
        self._act_dim = np.prod(training_environment.action_space.shape)
        self._model_type = model_type
        self._identity_terminal = identity_terminal
        self._hidden_dim = hidden_dim
        self._num_networks = num_networks
        self._num_elites = num_elites
        self._separate_mean_var = separate_mean_var
        self._model_name = model_name
        self._model_load_dir = model_load_dir
        self._deterministic = deterministic
        self._model = construct_model(obs_dim=self._obs_dim, act_dim=self._act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites,
                                      model_type=model_type, separate_mean_var=separate_mean_var,
                                      name=model_name, load_dir=model_load_dir,
                                      deterministic=deterministic, session=self._session)
        self._static_fns = static_fns

        self._rollout_schedule = [20, 100, rollout_length, rollout_length]
        self._max_model_t = max_model_t

        self._model_retain_epochs = model_retain_epochs
        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._rollout_random = rollout_random
        self._real_ratio = real_ratio
        self._Q_avgs = list()
        self._n_iters_qvar = [100]

        self._dynamic_risk_type = dynamic_risk
        self._dynamic_risk_param = None if dynamic_risk is 'neutral' else float(dynamic_risk.split('_')[1])
        self._rollout_repeats = rollout_repeats
        self._num_rollout_batches = num_rollout_batches

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = actor_lr
        self._Q_lr = critic_lr

        self._bc_lr = bc_lr
        self._bc_epochs = bc_epochs
        self._do_bc = pretrain_bc

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ 1R2R ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()
        self._state_samples = None
        self._batch_for_testing = None

        #### load replay pool data
        self._pool_load_path = pool_load_path

        obs_mean, obs_std = loader.restore_pool(
            self._pool,
            self._pool_load_path,
            save_path=self._log_dir,
            normalize_states=normalize_states,
            normalize_rewards=normalize_rewards,
            dataset_dir=dataset_dir
        )
        if normalize_states:
            self._obs_mean = obs_mean
            self._obs_std = obs_std
        else:
            self._obs_mean, self._obs_std = None, None

        self._init_pool_size = self._pool.size
        print('[ 1R2R ] Starting with pool size: {}'.format(self._init_pool_size))
        ####

        self.fake_env = FakeEnv(self._model, self._static_fns, penalty_coeff=0,
                                obs_mean=self._obs_mean, obs_std=self._obs_std)

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_bc_update()
        self._init_Q_target_no_entropy()

    def _train(self):
        """Return a generator that performs 1R2R offline RL training.
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        model_metrics = {}

        if not self._training_started:
            self._init_training()

        self.sampler.initialize(training_environment, policy, pool)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        if self._do_bc:
            print('[ 1R2R ] Behaviour cloning policy for {} epochs'.format(
                self._bc_epochs)
            )
            self._pretrain_bc(n_epochs=self._bc_epochs)

        #### model training
        print('[ 1R2R ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
        print('[ 1R2R ] Training model at epoch {} | freq {} | timestep {} (total: {})'.format(
            self._epoch, self._model_train_freq, self._timestep, self._total_timestep)
        )

        max_epochs = 1 if self._model.model_loaded else None
        model_train_metrics = self._train_model(
            batch_size=256,
            max_epochs=max_epochs,
            max_epochs_since_update=10,
            holdout_ratio=0.1,
            max_t=self._max_model_t
            )

        model_metrics.update(model_train_metrics)
        self._log_model()
        gt.stamp('epoch_train_model')
        ####

        if hasattr(self._training_environment.unwrapped, 'generate_plots'):
            self._training_environment.unwrapped.generate_plots(
                self._epoch,
                self.fake_env,
                self._Qs,
                self._policy
            )

        # main training loop
        for self._epoch in gt.timed_for(range(self._n_epochs)):
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            self._set_rollout_length()
            start_samples = self.sampler._total_samples

            for timestep in count():
                self._timestep = timestep

                if (timestep >= self._epoch_length
                    and self.ready_to_train):
                    break

                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                ## model rollouts
                if timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size, deterministic=self._deterministic)
                    model_metrics.update(model_rollout_metrics)

                    gt.stamp('epoch_rollout_model')

                ## train actor and critic
                if self.ready_to_train:
                    self._do_agent_training_repeats(timestep=timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))

            if self._epoch % self._evaluate_interval == 0 \
                or self._epoch >= self._n_epochs - self._avg_returns_num_iter:
                evaluation_paths = self._evaluation_paths(
                    policy,
                    evaluation_environment,
                    self._obs_mean,
                    self._obs_std
                )
                gt.stamp('evaluation_paths')

                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
            )))

            for iter in self._n_iters_qvar:
                diagnostics.update({
                    f'qvar/Q-var-{str(iter)}-iter': np.std(np.array(self._Q_avgs[max(self._epoch - iter, 0):]))**2
                })

            current_losses = self._model.validate()
            for i in range(len(current_losses)):
                diagnostics.update({'model/current_val_loss_' + str(i): current_losses[i]})
            diagnostics.update({'model/current_val_loss_avg': current_losses.mean()})

            self._writer.add_dict(diagnostics, self._epoch)
            for item in diagnostics.items():
                print(item)

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            ## ensure we did not collect any more data
            assert self._pool.size == self._init_pool_size

            if hasattr(self._training_environment.unwrapped, 'generate_plots'):
                self._training_environment.unwrapped.generate_plots(
                    self._epoch,
                    self.fake_env,
                    self._Qs,
                    self._policy
                )


            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        yield {'done': True, **diagnostics}


    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    def _log_model(self):
        if self._model_type == 'identity':
            print('[ 1R2R ] Identity model, skipping save')
        elif self._model.model_loaded:
            print('[ 1R2R ] Loaded model, skipping save')
        else:
            self._save_path = os.path.join(self._log_dir, 'models')
            filesystem.mkdir(self._save_path)
            print('[ 1R2R ] Saving model to: {}'.format(self._save_path))
            self._model.save(self._save_path, self._total_timestep)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ 1R2R ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ 1R2R ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        if self._model_type == 'identity':
            print('[ 1R2R ] Identity model, skipping model')
            model_metrics = {}
        else:
            env_samples = self._pool.return_all_samples()
            train_inputs, train_outputs = format_samples_for_training(env_samples)
            model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _pretrain_bc(self, batch_size=256, n_epochs=50, max_logging=2000, holdout_ratio=0.1):
        """ Pretrain the policy using behaviour cloning on the dataset
        """
        progress = Progress(n_epochs)

        env_samples = self._pool.return_all_samples()
        obs = env_samples["observations"]
        act = env_samples["actions"]

        # Split into training and holdout sets
        num_holdout = min(int(obs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(obs.shape[0])
        obs, holdout_obs = obs[permutation[num_holdout:]], obs[permutation[:num_holdout]]
        act, holdout_act = act[permutation[num_holdout:]], act[permutation[:num_holdout]]
        idxs = np.random.randint(obs.shape[0], size=[obs.shape[0]])

        mse_loss = self._session.run(
            self._mse_loss,
            feed_dict = {
                self._observations_ph: holdout_obs,
                self._actions_ph: holdout_act
            }
        )

        for i in range(n_epochs):
            for batch_num in range(int(obs.shape[0] // batch_size)):
                batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                acts = act[batch_idxs]
                obss = obs[batch_idxs]
                if np.max(acts) >= 1.0 or np.min(acts) <= -1.0:
                    continue

                feed_dict = {
                    self._observations_ph: obss,
                    self._actions_ph: acts
                }
                self._session.run(self._bc_train_op, feed_dict)

            mse_loss = self._session.run(
                self._mse_loss,
                feed_dict = {
                    self._observations_ph: holdout_obs,
                    self._actions_ph: holdout_act
                }
            )

            progress.update()
            progress.set_description([['bc_mse_loss', mse_loss]])

    def _rollout_model_neutral(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {} | Type: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size, self._model_type
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        steps_added = []
        for i in range(self._rollout_length):
            if not self._rollout_random:
                act = self._policy.actions_np(obs)
            else:
                act_ = self._policy.actions_np(obs)
                act = np.random.uniform(low=-1, high=1, size=act_.shape)

            if self._model_type == 'identity':
                next_obs = obs
                rew = np.zeros((len(obs), 1))
                term = (np.ones((len(obs), 1)) * self._identity_terminal).astype(np.bool)
                info = {}
            else:
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
            steps_added.append(len(obs))

            samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length,
                        'max_reward': np.max(rew),
                        'min_reward': np.min(rew),
                        'avg_reward': np.mean(rew),
                        'std_reward': np.std(rew)}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats


    def _rollout_model(self, rollout_batch_size, max_logging=1000, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size
        ))

        if self._dynamic_risk_type == 'neutral':
            return self._rollout_model_neutral(rollout_batch_size, **kwargs)

        assert ("cvar" in self._dynamic_risk_type or "wang" in self._dynamic_risk_type)

        steps_added = []
        batch = self.sampler.random_batch(rollout_batch_size)
        obs_all = batch['observations']
        rew_min, rew_max, rew_mean, rew_std = [], [], [], []

        for step in range(self._rollout_length):
            next_obs_all = []
            term_all = []

            # perform multiple batches of rollouts to reduce memory usage due to oversampling
            for i in range(self._num_rollout_batches):
                n_rollouts = obs_all.shape[0] // self._num_rollout_batches
                obs = obs_all[i*n_rollouts:(i + 1)*n_rollouts, :]

                if not self._rollout_random:
                    act = self._policy.actions_np(obs)
                else:
                    act_ = self._policy.actions_np(obs)
                    act = np.random.uniform(low=-1, high=1, size=act_.shape)

                # repeatedly sample the same state action pair
                obs_repeat = np.repeat(obs, self._rollout_repeats, axis=0)
                act_repeat = np.repeat(act, self._rollout_repeats, axis=0)

                next_obs, rew, term, info = self.fake_env.step(obs_repeat, act_repeat, **kwargs)

                batch = {'observations': obs_repeat, 'actions': act_repeat, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }

                # compute reward + next value
                feed_dict = self._get_feed_dict(iteration=None, batch=batch)
                Q_samples_flat = self._session.run(
                    (self._Q_estimates),
                    feed_dict
                )
                Q_samples = np.reshape(Q_samples_flat, (n_rollouts, -1, self._rollout_repeats))

                # sample the successor states to add to buffer according to risk measure.
                if "cvar" in self._dynamic_risk_type:
                    Q_samples_sorted = np.sort(Q_samples, axis=-1)
                    n_cvar_samples = int(math.ceil(self._dynamic_risk_param * self._rollout_repeats))
                    eps = 1e-4
                    var = Q_samples_sorted[:, :, n_cvar_samples - 1] + eps

                    less_than_var = Q_samples <= np.repeat(var[:, :, np.newaxis], self._rollout_repeats, axis=2)
                    less_than_var = np.reshape(less_than_var, (-1))

                    # subsample from those datapoints less than the var
                    args_less_than_var = np.argwhere(less_than_var).flatten()
                    row_indices = np.random.choice(args_less_than_var, size=n_rollouts, replace=False)

                elif "wang" in self._dynamic_risk_type:

                    # compute reweighted discrete distribution according to wang distortion
                    approx_cdf = np.sort(np.linspace(1, 0, num=self._rollout_repeats, endpoint=False))
                    new_cdf = wang_cdf(approx_cdf, self._dynamic_risk_param)
                    dist = new_cdf - np.insert(new_cdf, 0, 0.)[:-1]

                    # sample according to wang distortion
                    sampled_indices_ordered = np.random.choice(range(self._rollout_repeats), size=n_rollouts, p=dist)

                    # order indices from worst to best successor
                    Q_samples_argsort = np.squeeze(np.argsort(Q_samples, axis=-1))

                    # retrieve the sampled indices
                    Q_samples_indices = Q_samples_argsort[np.arange(len(Q_samples_argsort)), sampled_indices_ordered]

                    # prepare to index flat arrays
                    row_indices = Q_samples_indices + np.arange(len(Q_samples_indices)) * self._rollout_repeats

                obs = obs_repeat[row_indices, :]
                act = act_repeat[row_indices, :]
                next_obs = next_obs[row_indices, :]
                rew = rew[row_indices, :]
                term = term[row_indices, :]
                samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
                self._model_pool.add_samples(samples)
                steps_added.append(rew.shape[0])

                # append for next step
                next_obs_all.append(next_obs)
                term_all.append(term)

            # gather all of the states from which to initiate next step
            next_obs_all = np.concatenate(next_obs_all, axis=0)
            term_all = np.concatenate(term_all, axis=0)
            nonterm_mask = ~term_all.squeeze(-1)

            if nonterm_mask.sum() == 0:
                print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs_all = next_obs_all[nonterm_mask]

            rew_min.append(np.min(rew[:max_logging]))
            rew_max.append(np.max(rew[:max_logging]))
            rew_mean.append(np.mean(rew[:max_logging]))
            rew_std.append(np.std(rew[:max_logging]))

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {
            'mean_rollout_length': mean_rollout_length,
            'reward_min': sum(rew_min) / self._rollout_length,
            'reward_max': sum(rew_max) / self._rollout_length,
            'reward_mean': sum(rew_mean) / self._rollout_length,
            'reward_std': sum(rew_std) / self._rollout_length,
        }

        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        env_batch = self._pool.random_batch(env_batch_size)

        obs = env_batch["observations"]
        next_obs = env_batch["next_observations"]
        deltas = next_obs - obs

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            # keys = env_batch.keys()
            keys = set(env_batch.keys()) & set(model_batch.keys())
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def _real_data_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch = self._pool.random_batch(batch_size)
        return env_batch

    def _compare_policy_to_data(self):
        """
        Compute the mean squared error between the actions taken in the dataset
        and the actions taken under the current policy.

        Returns:
            mae_per_action: the mean absolute error for each action dimension
            of the policy actions versus the dataset actions.
        """
        env_batch = self._real_data_batch()
        observations = env_batch["observations"]
        dataset_actions = env_batch["actions"]
        with self._policy.set_deterministic(True):
            policy_actions = self._policy.actions_np(observations)
        mae_per_action = (np.abs(dataset_actions - policy_actions)).mean(axis=0)
        return mae_per_action.tolist()

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target


    def _init_Q_target_no_entropy(self):
        """ Estimate the reward plus the future expected value for
        some transitions.

        This excludes the entropy bonus and uses Q rather than Q targ.
        """
        next_actions = self._policy.actions([self._next_observations_ph])
        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Qs)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q

        Q_target = self._Q_estimates = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

    def _init_bc_update(self):
        """ Initialise update to initially perform behaviour cloning on the
        dataset prior to running 1R2R.
        """
        log_pis = self._policy.log_pis([self._observations_ph], self._actions_ph)
        bc_loss = self._bc_loss = -tf.reduce_mean(log_pis)

        actions = self._policy.actions([self._observations_ph])
        mse = self._mse_loss = tf.reduce_mean(tf.square(actions - self._actions_ph))

        self._bc_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._bc_lr,
            name="bc_optimizer")
        bc_train_op = tf.contrib.layers.optimize_loss(
            bc_loss,
            self.global_step,
            learning_rate=self._bc_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._bc_train_op = bc_train_op

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_agent_training_repeats(self, timestep):
        """Repeat training _n_train_repeat times every _train_every_n_steps"""
        if timestep % self._train_every_n_steps > 0: return
        trained_enough = (
            self._train_steps_this_epoch
            > self._max_train_repeat_per_timestep * self._timestep)
        if trained_enough: return

        for i in range(self._n_train_repeat):
            self._train_agent(
                iteration=timestep,
                batch=self._training_batch())

        self._num_train_steps += self._n_train_repeat
        self._train_steps_this_epoch += self._n_train_repeat

    def _train_agent(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             self.global_step),
            feed_dict)
        self._Q_avgs.append(np.mean(Q_values))

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha
        })

        # TODO : Remove
        if np.abs(np.mean(Q_values)) > 1e10:
            sys.exit(0)

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        policy_mae = self._compare_policy_to_data()
        diagnostics.update({
            'policy/dataset_mae_action_avg': np.mean(policy_mae)
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
