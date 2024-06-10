import numpy as np
import pandapipes

from gymnasium import spaces
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

from mo_gymnasium.envs.pandapipes.network import GasNetwork
from mo_gymnasium.envs.pandapipes.controller import GasNetworkController
from mo_gymnasium.envs.pandapipes.visualization import GasNetworkVisualizer

class MOHopperEnv(HopperEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HopperEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    The original Gymnasium's 'Hopper-v4' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-hopper-v4', cost_objective=False)
    LinearReward(env, weight=np.array([1.0, 0.0]))

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    """

    def __init__(self, cost_objective=True, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, cost_objective, **kwargs)
        self.cost_objetive = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))

        ################
        # GasGridMOEnv #
        ################
        """
        todo(lisca): Add documentation string.
        """
        self.nO = 2  # nO
        self._verbose = False  # verbose

        self._gas_network = GasNetwork()

        self._gas_network_controller = \
            GasNetworkController(self._gas_network, verbose=self._verbose)

        self._gas_network_visualizer = \
            GasNetworkVisualizer(self._gas_network, verbose=self._verbose)

        self._time_step = 0

        # Simulate one step to initialize the OutputWriter, which will be queried for
        # the first observation.
        # Warning: without this step the computation of the action and observation
        #          spaces will fail.
        self._gas_network_controller._controller_storage.reset()
        pandapipes.timeseries.run_timeseries(
            self._gas_network, time_steps=range(1), verbose=self._verbose)

        # self.action_space      = self._define_action_space()
        # self.observation_space = self._define_observation_space()
        # self.reward_space      = self._define_reward_space()

        self._action_previous  = None    

    def _define_action_space(self):
        """
        todo(lisca): Add documentation string.
        """
        action_box = spaces.Discrete(2)

        # Note: the action is directly defined.
        # action_box = gymnasium.spaces.Box(low=-0.02, high=0.02, shape=(1,))

        return action_box

    def _define_observation_space(self):
        """
        todo(lisca): Add documentation string.
        """
        observation_box = spaces.Box(low=0, high=np.inf, shape=(1,))

        # # Note: the observation is indirectly defined, by taking the size of the array
        # # which is assembled from the variables of the OutputWriter.
        # # We need a dummy action, just ran
        # observation_sample = self._get_obs()

        # observation_box = gymnasium.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(observation_sample.shape[0],))

        return observation_box

    def _define_reward_space(self):
        """
        todo(lisca): Add documentation string.
        """
        reward_box = spaces.Box(-np.inf, np.inf, shape=(self.nO,))

        # # Note: Defines which of the first nO rewards will be accounted for.
        # reward_box = gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.nO,))

        return reward_box

    def _get_obs(self, action=None):
        """
        todo(lisca): Add documentation string.
        """

        # Use Hopper's _get_obs() method.
        return super()._get_obs()

        # observation = np.array([
        #     # ## Time
        #     # self._time_step,
        #     # ## Flows
        #     # self._gas_network_controller._output_writer.np_results['source.mdot_kg_per_s'][0, 0],
        #     # self._gas_network_controller._output_writer.np_results['sink.mdot_kg_per_s'][0, 0],
        #     # self._gas_network_controller._output_writer.np_results['res_ext_grid.mdot_kg_per_s'][0, 0],
        #     # self._gas_network_controller._output_writer.np_results['res_mass_storage.mdot_kg_per_s'][0, 0],
        #     ## Quantities
        #     self._gas_network_controller._output_writer.np_results['mass_storage.m_stored_kg'][0, 0],], dtype=np.float32)
        # return observation

    def _storage_reward(self, x):
        """
        todo(lisca): Add documentation string.
        """
        # Storage should be between 25% and 75%!
        if 0 <= x < 0.25:
            return 0
        elif 0.25 <= x <= 0.5:
            return 4 * (x - 0.25)
        elif 0.5 < x < 0.75:
            return 1 - 4 * (x - 0.5)
        elif 0.75 <= x <= 1:
            return 0

    def _get_reward(self, action_current=None):
        """
        todo(lisca): Add documentation string.
        """
        # r0. Keep the storage between 25% and 75%!
        filling_level_percent = self._gas_network_controller._output_writer.np_results['mass_storage.filling_level_percent'][0, 0]
        # Convert the percentage to [0,1], compute the reward, and convert back to kg.
        # Note: - the factor * 10000 is only for numerical alignment with PandapipesMOEnv.
        reward_storage = self._storage_reward(filling_level_percent / 100) * 10000
        # print(f"r0: {filling_level_percent} -> {reward_storage}")

        # r1. Keep m dot as constant as possible (m dot dot close to 0, ideally 0)!
        # Note: - the factor *= 100 is used to bring the action into the interval [0, 10]
        #       - the factor *3.0 is used only for numerical alignment with PandapipesMOEnv.
        if self._time_step > 0:
            difference = abs(action_current - self._action_previous)
            difference *= 100
            # We assume that we have only one action: value of m dot.
            reward_difference = -np.exp(difference)[0] * 3.0
            # print(f"r1: {difference} -> {reward_difference}")
        else:
            # The action after reset has no action previous to it,
            # therefore we can no compute this reward.
            reward_difference = 0

        # r2. Minimize external grid mass flow:
        # - should neither be positive or negative
        # - ideally just work with charging/discharging storage
        # Note: - the factor *100000 is only for numerical alignment with PandapipesMOEnv.
        flow_external_grid = self._gas_network_controller._output_writer.np_results['res_ext_grid.mdot_kg_per_s'][0, 0]
        reward_external_grid = -np.abs(flow_external_grid) * 100000
        # print(f"r2: {flow_external_grid} -> {reward_external_grid}")

        r0 = reward_storage
        r1 = reward_difference
        r2 = reward_external_grid
        r3 = 0.0

        reward = np.array([r0, r1, r2, r3], dtype=np.float64)[:self.nO].flatten()
        return reward

    def _get_info(self):
        """
        todo(lisca): Add documentation string.
        """

        info = {}
        return info

    def reset(self, seed=None, options=None):
        """
        todo(lisca): Add documentation string.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = {
            "x_position": self.init_qpos[0],
            "x_velocity": 0.0,
            "height_reward": self.init_qpos[1],
            "energy_reward": 0.0,
        }

        ################
        # GasGridMOEnv #
        ################
        # Reset the pandapipes simulation.
        self._gas_network_controller._controller_storage.reset()

        # Simulate one step to initialize the OutputWriter. Without it
        # the query for 1st observation will fail!
        pandapipes.timeseries.run_timeseries(
            self._gas_network, time_steps=range(1), verbose=False)

        self._time_step = 0

        # observation = self._get_obs()
        # info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # # print(f'\u001b[33m{self._time_step:>3} -> reset\u001b[0m')

        return observation, info    
    
    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        # costs = ctrl_cost

        observation = self._get_obs()
        # reward = rewards - costs
        terminated = self.terminated

        z = self.data.qpos[1]
        height = 10 * (z - self.init_qpos[1])
        energy_cost = np.sum(np.square(action))

        if self.cost_objetive:
            vec_reward = np.array([x_velocity, height, -energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward -= self._ctrl_cost_weight * energy_cost

        vec_reward += healthy_reward

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "height_reward": height,
            "energy_reward": -energy_cost,
        }

        if self.render_mode == "human":
            self.render()

        ################
        # GasGridMOEnv #
        ################
        """
        todo(lisca): Add documentation string.
        """

        # todo(lisca): remove this hack!
        action = 0.001  # action / 1000.0

        # Set the new mdot_kg_per_s of the storage controller.
        self._gas_network_controller._controller_storage.mdot_kg_per_s = action

        # ? stochastic (m_dot of the) sink

        # Simulate one hour.
        time_steps = range(self._time_step, self._time_step + 1)
        pandapipes.timeseries.run_timeseries(
            self._gas_network, time_steps=time_steps, verbose=False)

        # observation = self._get_obs(action)
        # reward      = self._get_reward(action)
        # terminated  = False
        # truncated   = False
        # info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # if self._time_step % 1 == 0:
        #     print(f'\u001b[31m{self._time_step:>3} -> action: {action}\u001b[0m')
        #     print(f'\u001b[32m{self._time_step:>3} -> reward: {reward}\u001b[0m')

        self._action_previous = action
        self._time_step += 1        
        
        return observation, vec_reward, terminated, False, info
