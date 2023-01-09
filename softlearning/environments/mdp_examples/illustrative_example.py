import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib
from gym import spaces
from datetime import datetime
import os
dirname = os.path.dirname(__file__)

matplotlib.rcParams.update({'font.size': 22})

def normalize(obs, mean, std):
    obs_norm = (obs - mean) / std
    return obs_norm

class IllustrativeExample(gym.Env):
    def __init__(self):
        self.n_samples = 5000

        self.obs_low = np.array([-10])
        self.obs_high = np.array([10])
        act_low = np.array([-1])
        act_high = np.array([1])
        obs_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.observation_space = spaces.Dict({"observation": obs_space})
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.active_observation_shape = obs_space.shape

        self.init_s = -1
        self.s_current = 0.
        self.plot_save_path = None

        self.dataset_plotted = False
        self.model_plotted = False

    def create_plot_folder(self):
        dirname = os.path.dirname(__file__)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%H-%M-%S")
        self.plot_save_path = os.path.join(dirname, 'plots/multimodal' + dt_string)
        os.mkdir(self.plot_save_path)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.s_current = self.init_s
        return np.array([self.s_current])

    def step(self, action):
        next_state_noise = 0.2
        reward_noise = 0.2
        if isinstance(action, np.ndarray):
            action = action[0]
        s_next_mean = action ** 2 + 1
        s_next_std = max(0.001, 0.25*(action + 0.4))
        if self.s_current > self.init_s + 0.1:
            return np.array([self.s_current]), self.s_current, True, {}

        else:
            reward = 0.0
            self.s_current = s_next_mean + np.random.normal(loc=0, scale=s_next_std)
            return np.array([self.s_current]), reward, False, {}

    def generate_plots(self, epoch, fake_env, Qs, policy):
        if self.plot_save_path is None:
            self.create_plot_folder()

        if not self.dataset_plotted:
            self.generate_dataset()
            self.plot_dataset()
            self.dataset_plotted = True

        if not self.model_plotted:
            self.plot_model(epoch, fake_env)
            self.plot_rewards(epoch, fake_env)
            self.model_plotted = True

        self.plot_Q(epoch, Qs, policy)
        self.plot_optimal_action_value(epoch, Qs, policy)

    def plot_optimal_action_value(self, epoch, Qs, policy):
        points = 1000
        states = np.expand_dims(np.linspace(start=-1, stop=5, num=points), 1)

        with policy.set_deterministic(True):
            opt_actions = policy.actions_np(states)

        q1 = Qs[0].predict([states, opt_actions])
        q2 = Qs[1].predict([states, opt_actions])
        opt_q = np.minimum(q1, q2)

        plt.figure()
        plt.plot(states, opt_q, "kx")
        plt.xlim([-1, 5])
        plt.ylim([-2, 10])
        plt.title("Optimal values as function of s")
        plt.xlabel("s")
        plt.ylabel("V")
        plt.grid()
        plt.savefig(os.path.join(self.plot_save_path, 'state_values_at_epoch_' + str(epoch) + ".png"), bbox_inches='tight',dpi=200)
        plt.close()

    def plot_model(self, epoch, fake_env):
        points = 1000
        states = np.zeros((points, 1))
        actions = np.linspace(start=-1, stop=1, num=points)
        actions = np.expand_dims(actions, 1)

        num_samples = 20
        next_state_samples = np.zeros((points, 1, num_samples))
        for i in range(num_samples):
            next_states, rewards, terminals, _ = fake_env.step(states, actions)
            next_state_samples[:, :, i] = next_states

        next_state_means = np.mean(next_state_samples, axis=2)
        next_state_stds = np.std(next_state_samples, axis=2)

        # do some smoothing for a better plot
        window_width = 50
        next_means_cumsum = np.cumsum(np.insert(next_state_means[:, 0], 0, 0))
        next_stds_cumsum = np.cumsum(np.insert(next_state_stds[:, 0], 0, 0))
        acts_cum_sum = np.cumsum(np.insert(actions[:, 0], 0, 0))
        next_means_ma = (next_means_cumsum[window_width:] - next_means_cumsum[:-window_width]) / window_width
        next_stds_ma = (next_stds_cumsum[window_width:] - next_stds_cumsum[:-window_width]) / window_width
        acts_ma = (acts_cum_sum[window_width:] - acts_cum_sum[:-window_width]) / window_width

        plt.figure()
        plt.fill_between(acts_ma, next_means_ma + next_stds_ma, next_means_ma - next_stds_ma, alpha=0.3, color="navy")
        plt.plot(acts_ma, next_means_ma, color="navy")
        plt.xlabel(r"$a$")
        plt.ylabel(r"$\widehat{T}(s'\ |\ a, s = 0)$")
        plt.xlim([-1, 1])
        plt.ylim([0, 5])
        plt.grid()
        plt.savefig(os.path.join(self.plot_save_path, 'model_at_epoch_' + str(epoch) + ".png"), bbox_inches='tight', dpi=200)


    def plot_rewards(self, epoch, fake_env):
        points = 2000
        states = np.expand_dims(np.linspace(start=0, stop=10, num=points), 1)
        actions = np.zeros((points, 1))

        num_samples = 20
        reward_samples = np.zeros((points, 1, num_samples))
        for i in range(num_samples):
            next_states, rewards, terminals, _ = fake_env.step(states, actions)
            reward_samples[:, :, i] = rewards

        reward_means = np.mean(reward_samples, axis=2)
        reward_stds = np.std(reward_samples, axis=2)
        plt.figure()
        plt.title("Reward function")
        plt.xlabel("s_1")
        plt.ylabel("reward")
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.grid()
        plt.fill_between(states[:, 0], reward_means[:, 0] + reward_stds[:, 0], reward_means[:, 0] - reward_stds[:, 0])
        plt.savefig(os.path.join(self.plot_save_path, 'reward_model_at_epoch_' + str(epoch) + ".png"), bbox_inches='tight', dpi=200)
        plt.xlabel("")

    def plot_Q(self, epoch, Qs, policy):
        points = 2000
        states = np.zeros((points, 1))
        actions = np.linspace(start=-1, stop=1, num=points)
        actions = np.expand_dims(actions, 1)

        q1 = Qs[0].predict([states, actions])
        q2 = Qs[1].predict([states, actions])
        q = np.minimum(q1, q2)

        state = np.zeros((1, 1))
        with policy.set_deterministic(True):
            opt_actions = policy.actions_np(state)

        plt.figure()
        plt.plot(actions, q, "k", linewidth=3)
        plt.xlim([-1, 1])
        plt.ylim([-1, 3])
        plt.xlabel(r"$a$")
        plt.ylabel(r"$Q(s = 0, a)$")
        plt.grid()
        plt.axvline(x=opt_actions[0][0], color='red', linewidth=4.)
        plt.savefig(os.path.join(self.plot_save_path, 'q_values_at_epoch_' + str(epoch) + ".png"), bbox_inches='tight', dpi=200)

    def plot_dataset(self):
        if self.plot_save_path is None:
            self.create_plot_folder()

        states = self.dataset['observations'][:, 0]
        actions = self.dataset['actions']
        next_states = self.dataset['next_observations'][:, 0]

        init_states = states[states < 0.01]
        init_actions = actions[states < 0.01]
        init_next_states = next_states[states < 0.01]
        plt.figure()
        plt.plot(init_actions, init_next_states, "ko", markersize=1)
        plt.xlabel(r"$a$")
        plt.ylabel(r"$s'$")
        plt.xlim([-1, 1])
        plt.ylim([0, 2])
        plt.grid()
        plt.savefig(os.path.join(self.plot_save_path, 'state_data.png'), bbox_inches='tight', dpi=200)

        states = self.dataset['observations'][:, 0]
        rewards = self.dataset['rewards']
        plt.figure()
        plt.xlabel("s_1")
        plt.ylabel("reward")
        plt.xlim([0, 5])
        plt.ylim([0, 5])
        plt.grid()
        plt.plot(states, rewards[:, 0], "ko")
        plt.savefig(os.path.join(self.plot_save_path, 'reward_data.png'), bbox_inches='tight', dpi=200)

    def compute_normalisation_factors(self):
        obs = self.dataset["observations"]
        self.obs_mean = np.mean(obs, axis=0)
        self.obs_std = np.std(obs, axis=0)

    def generate_dataset(self):
        rewards = np.zeros((self.n_samples, 1))
        observations = np.zeros((self.n_samples, 1))
        next_observations = np.zeros((self.n_samples, 1))
        actions = np.zeros((self.n_samples, 1))
        terminals = np.full((self.n_samples, 1), False, dtype=bool)

        step = 0
        while True:

            state = self.reset()[0]
            action = np.random.uniform(low=-0.3, high=0.4)

            next_state, reward, terminal, info = self.step(action)
            next_state = next_state[0]
            if step < self.n_samples:
                rewards[step, 0] = reward
                observations[step, 0] = state
                next_observations[step, 0] = next_state
                terminals[step, 0] = terminal
                actions[step, 0] = action
                step += 1

            state = next_state
            action = np.random.uniform(low=-1, high=1)
            next_state, reward, terminal, info = self.step(action)

            if step < self.n_samples:
                rewards[step, 0] = reward
                observations[step, 0] = state
                next_observations[step, 0] = next_state
                terminals[step, 0] = terminal
                actions[step, 0] = action
                step += 1

            if step >= self.n_samples:
                dataset = {
                    'observations': observations,
                    'actions': actions,
                    'next_observations': next_observations,
                    'rewards': rewards,
                    'terminals': terminals
                    }

                self.dataset = dataset

                return dataset
