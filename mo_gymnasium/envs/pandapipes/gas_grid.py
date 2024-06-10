import gymnasium
import math
import numpy as np
import pandapipes

from envs.pandapipes.network import GasNetwork
from envs.pandapipes.controller import GasNetworkController
from envs.pandapipes.visualization import GasNetworkVisualizer

# from gymnasium.envs.classic_control.mountain_car import MountainCarEnv  # to be deleted after inheriting direct from gymnasium.Env, instead of MountainCarEnv, which inherits from gymnasium.Env=
from gymnasium import spaces
from gymnasium.utils import EzPickle
from typing import List, Optional

from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

# import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces.box import Box
from gymnasium.utils import EzPickle


class PandapipesMOEnv(gymnasium.Env, EzPickle):
    """
    ## Description
    A Water reservoir environment.
    The agent executes a continuous action, corresponding to the amount of water released by the dam.

    A. Castelletti, F. Pianosi and M. Restelli, "Tree-based Fitted Q-iteration for Multi-Objective Markov Decision problems,"
    The 2012 International Joint Conference on Neural Networks (IJCNN),
    Brisbane, QLD, Australia, 2012, pp. 1-8, doi: 10.1109/IJCNN.2012.6252759.

    ## Observation Space
    The observation is a float corresponding to the current level of the reservoir.

    ## Action Space
    The action is a float corresponding to the amount of water released by the dam.
    If normalized_action is True, the action is a float between 0 and 1 corresponding to the percentage of water released by the dam.

    ## Reward Space
    There are up to 4 rewards:
     - cost due to excess level wrt a flooding threshold (upstream)
     - deficit in the water supply wrt the water demand
     - deficit in hydroelectric supply wrt hydroelectric demand
     - cost due to excess level wrt a flooding threshold (downstream)
     By default, only the first two are used.

     ## Starting State
     The reservoir is initialized with a random level between 0 and 160.

     ## Arguments
        - render_mode: The render mode to use. Can be 'human', 'rgb_array' or 'ansi'.
        - time_limit: The maximum number of steps until the episode is truncated.
        - nO: The number of objectives to use. Can be 2, 3 or 4.
        - penalize: Whether to penalize the agent for selecting an action out of bounds.
        - normalized_action: Whether to normalize the action space as a percentage [0, 1].
        - initial_state: The initial state of the reservoir. If None, a random state is used.

     ## Credits
     Code from:
     [Mathieu Reymond](https://gitlab.ai.vub.ac.be/mreymond/dam).
     Ported from:
     [Simone Parisi](https://github.com/sparisi/mips).

     Sky background image from: Paulina Riva (https://opengameart.org/content/sky-background)
    """

    S = 1.0  # Reservoir surface
    W_IRR = 50.0  # Water demand
    H_FLO_U = 50.0  # Flooding threshold (upstream, i.e. height of dam)
    S_MIN_REL = 100.0  # Release threshold (i.e. max capacity)
    DAM_INFLOW_MEAN = 40.0  # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 10.0
    Q_MEF = 0.0
    GAMMA_H2O = 1000.0  # water density
    W_HYD = 4.36  # Hydroelectric demand
    Q_FLO_D = 30.0  # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1.0  # Turbine efficiency
    G = 9.81  # Gravity

    utopia = {2: [-0.5, -9], 3: [-0.5, -9, -0.0001], 4: [-0.5, -9, -0.001, -9]}
    antiutopia = {2: [-2.5, -11], 3: [-65, -12, -0.7], 4: [-65, -12, -0.7, -12]}

    # Create colors.
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    s_init = np.array(
        [
            9.6855361e01,
            5.8046026e01,
            1.1615767e02,
            2.0164311e01,
            7.9191000e01,
            1.4013098e02,
            1.3101816e02,
            4.4351321e01,
            1.3185943e01,
            7.3508622e01,
        ],
        dtype=np.float32,
    )

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 2}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_limit: int = 100,
        nO=2,
        penalize: bool = False,
        normalized_action: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ):
        EzPickle.__init__(self, render_mode, time_limit, nO, penalize, normalized_action)
        self.render_mode = render_mode

        # self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        # PandapipesMOEnv
        self.observation_space = Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
        self.normalized_action = normalized_action
        if self.normalized_action:
            self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # self.action_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
            # PandapipesMOEnv
            # -> 1000.0 as high is just a guess.
            self.action_space = Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)

        self.nO = nO
        self.penalize = penalize
        self.time_limit = time_limit
        self.initial_state = initial_state
        self.time_step = 0
        self.last_action = None
        self.dam_inflow = None
        self.excess = None
        self.defict = None

        low = -np.ones(nO) * np.inf  # PandapipesMOEnv.antiutopia[nO]
        high = np.zeros(nO)  # PandapipesMOEnv.utopia[nO]
        self.reward_space = Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.reward_dim = nO

        self.window = None
        self.window_size = (300, 200)  # width x height
        self.clock = None
        self.water_img = None
        self.wall_img = None
        self.sky_img = None

    def pareto_front(self, gamma: float) -> List[np.ndarray]:
        """This function returns the pareto front of the resource gathering environment.

        Args:
            gamma (float): The discount factor.

        Returns:
            The pareto front of the resource gathering environment.
        """

        def get_non_dominated(candidates: List[np.ndarray]) -> List[np.ndarray]:
            """This function returns the non-dominated subset of elements.

            Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
            The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.

            Args:
                candidates: The input set of candidate vectors.

            Returns:
                The non-dominated subset of this input set.
            """
            candidates = np.array(candidates)  # Turn the input set into a numpy array.
            candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
            for i in range(candidates.shape[0]):  # Process each point in turn.
                n = candidates.shape[0]  # Check current size of the candidates.
                if i >= n:  # If we've eliminated everything up until this size we stop.
                    break
                non_dominated = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
                # find all points not dominated by i
                # since points are sorted by coordinate sum
                # i cannot dominate any points in 1,...,i-1
                non_dominated[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
                candidates = candidates[non_dominated]  # Grab only the non-dominated vectors using the generated bitmask.

            non_dominated = set()
            for candidate in candidates:
                non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.

            return [np.array(point) for point in non_dominated]

        # Go directly to the diamond (R2) in 10 steps
        ret1 = np.array([0.0, 0.0, 1.0]) * gamma**10

        # Go to both resources, through both Es
        ret2 = 0.9 * 0.9 * np.array([0.0, 1.0, 1.0]) * gamma**12  # Didn't die
        ret2 += 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**7  # Died to E2
        ret2 += 0.9 * 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**9  # Died to E1

        # Go to gold (R1), through E1 both ways
        ret3 = 0.9 * 0.9 * np.array([0.0, 1.0, 0.0]) * gamma**8  # Didn't die
        ret3 += 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**3  # Died to E1
        ret3 += 0.9 * 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**5  # Died to E1 in the way back

        # Go to both resources, dodging E1 but through E2
        ret4 = 0.9 * np.array([0.0, 1.0, 1.0]) * gamma**14  # Didn't die
        ret4 += 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**7  # Died to E2

        # Go to gold (R1), doging all E's in 12 steps
        ret5 = np.array([0.0, 1.0, 0.0]) * gamma**12  # Didn't die

        # Go to gold (R1), going through E1 only once
        ret6 = 0.9 * np.array([0.0, 1.0, 0.0]) * gamma**10  # Didn't die
        ret6 += 0.1 * np.array([-1.0, 0.0, 0.0]) * gamma**7  # Died to E1

        return get_non_dominated([ret1, ret2, ret3, ret4, ret5, ret6])
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time_step = 0
        if self.initial_state is not None:
            state = self.initial_state
        else:
            if not self.penalize:
                state = self.np_random.choice(PandapipesMOEnv.s_init, size=1)
            else:
                state = self.np_random.integers(0, 160, size=1)

        self.state = np.array(state, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. mo_gymnasium.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, render_mode: str):
        if self.window is None:
            pygame.init()

            if render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Water Reservoir")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if self.water_img is None:
                self.water_img = pygame.image.load(path.join(path.dirname(__file__), "assets/water.png"))
            if self.wall_img is None:
                self.wall_img = pygame.image.load(path.join(path.dirname(__file__), "assets/wall.png"))
            if self.sky_img is None:
                self.sky_img = pygame.image.load(path.join(path.dirname(__file__), "assets/sky.png"))
                self.sky_img = pygame.transform.flip(self.sky_img, False, True)
                self.sky_img = pygame.transform.scale(self.sky_img, self.window_size)

            self.font = pygame.font.Font(path.join(path.dirname(__file__), "assets", "Minecraft.ttf"), 15)

        self.window.blit(self.sky_img, (0, 0))

        # Draw the dam
        for x in range(self.wall_img.get_width(), self.window_size[0] - self.wall_img.get_width(), self.water_img.get_width()):
            for y in range(self.window_size[1] - int(self.state[0]), self.window_size[1], self.water_img.get_height()):
                self.window.blit(self.water_img, (x, y))

        # Draw the wall
        for y in range(0, int(PandapipesMOEnv.H_FLO_U), self.wall_img.get_width()):
            self.window.blit(self.wall_img, (0, self.window_size[1] - y - self.wall_img.get_height()))
            self.window.blit(
                self.wall_img,
                (self.window_size[0] - self.wall_img.get_width(), self.window_size[1] - y - self.wall_img.get_height()),
            )

        if self.last_action is not None:
            img = self.font.render(f"Water Released: {self.last_action:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 10))
            img = self.font.render(f"Dam Inflow: {self.dam_inflow:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 25))
            img = self.font.render(f"Water Level: {self.state[0]:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 40))
            img = self.font.render(f"Demand Deficit: {self.defict:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 55))
            img = self.font.render(f"Flooding Excess: {self.excess:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 70))

        img = self.font.render("Flooding threshold", True, (255, 0, 0))
        self.window.blit(img, (20, self.window_size[1] - PandapipesMOEnv.H_FLO_U))
        pygame.draw.line(
            self.window,
            (255, 0, 0),
            (0, self.window_size[1] - PandapipesMOEnv.H_FLO_U),
            (self.window_size[0], self.window_size[1] - PandapipesMOEnv.H_FLO_U),
        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _render_text(self):
        outfile = StringIO()
        outfile.write(f"Water level: {self.state[0]:.2f}\n")
        if self.last_action is not None:
            outfile.write(f"Water released: {self.last_action:.2f}\n")
            outfile.write(f"Dam inflow: {self.dam_inflow:.2f}\n")
            outfile.write(f"Demand deficit: {self.defict:.2f}\n")
            outfile.write(f"Flooding excess: {self.excess:.2f}\n")

        with closing(outfile):
            return outfile.getvalue()

    def step(self, action):
        # bound the action
        actionLB = np.clip(self.state - PandapipesMOEnv.S_MIN_REL, 0, None)
        actionUB = self.state

        if self.normalized_action:
            action = action * (actionUB - actionLB) + actionLB
            penalty = 0.0
        else:
            # Penalty proportional to the violation
            bounded_action = np.clip(action, actionLB, actionUB)
            penalty = -self.penalize * np.abs(bounded_action - action)
            action = bounded_action

        # transition dynamic
        self.last_action = action[0]
        self.dam_inflow = self.np_random.normal(PandapipesMOEnv.DAM_INFLOW_MEAN, PandapipesMOEnv.DAM_INFLOW_STD, len(self.state))[0]
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + self.dam_inflow - action, 0, None).astype(np.float32)

        # cost due to excess level wrt a flooding threshold (upstream)
        self.excess = np.clip(n_state / PandapipesMOEnv.S - PandapipesMOEnv.H_FLO_U, 0, None)[0]
        r0 = -self.excess + penalty
        # deficit in the water supply wrt the water demand
        self.defict = -np.clip(PandapipesMOEnv.W_IRR - action, 0, None)[0]
        r1 = self.defict + penalty

        q = np.clip(action[0] - PandapipesMOEnv.Q_MEF, 0, None)
        p_hyd = PandapipesMOEnv.ETA * PandapipesMOEnv.G * PandapipesMOEnv.GAMMA_H2O * n_state[0] / PandapipesMOEnv.S * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(PandapipesMOEnv.W_HYD - p_hyd, 0, None) + penalty
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(action[0] - PandapipesMOEnv.Q_FLO_D, 0, None) + penalty

        reward = np.array([r0, r1, r2, r3], dtype=np.float32)[: self.nO].flatten()

        self.state = n_state

        self.time_step += 1
        truncated = self.time_step >= self.time_limit
        terminated = False

        if self.render_mode == "human":
            self.render()

        return n_state, reward, terminated, truncated, {}


class PandapipesEnv(gymnasium.Env):
    """
    todo(lisca): Add documentation string.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, verbose=False, nO=2, penalize=False):
        super(PandapipesEnv, self).__init__()
        """
        todo(lisca): Add documentation string.
        """
        self.nO = nO
        self._verbose = verbose

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

        self.action_space      = self._define_action_space()
        self.observation_space = self._define_observation_space()
        self.reward_space      = self._define_reward_space()

        self._action_previous  = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _define_action_space(self):
        """
        todo(lisca): Add documentation string.
        """
        action_box = gymnasium.spaces.Discrete(2)

        # Note: the action is directly defined.
        # action_box = gymnasium.spaces.Box(low=-0.02, high=0.02, shape=(1,))

        return action_box

    def _define_observation_space(self):
        """
        todo(lisca): Add documentation string.
        """
        observation_box = gymnasium.spaces.Box(low=0, high=np.inf, shape=(1,))

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
        reward_box = gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.nO,))

        # # Note: Defines which of the first nO rewards will be accounted for.
        # reward_box = gymnasium.spaces.Box(-np.inf, np.inf, shape=(self.nO,))

        return reward_box

    def _get_obs(self, action=None):
        """
        todo(lisca): Add documentation string.
        """

        observation = np.array([
            # ## Time
            # self._time_step,
            # ## Flows
            # self._gas_network_controller._output_writer.np_results['source.mdot_kg_per_s'][0, 0],
            # self._gas_network_controller._output_writer.np_results['sink.mdot_kg_per_s'][0, 0],
            # self._gas_network_controller._output_writer.np_results['res_ext_grid.mdot_kg_per_s'][0, 0],
            # self._gas_network_controller._output_writer.np_results['res_mass_storage.mdot_kg_per_s'][0, 0],
            ## Quantities
            self._gas_network_controller._output_writer.np_results['mass_storage.m_stored_kg'][0, 0],], dtype=np.float32)

        return observation

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

        # Reset the pandapipes simulation.
        self._gas_network_controller._controller_storage.reset()

        # Simulate one step to initialize the OutputWriter. Without it
        # the query for 1st observation will fail!
        pandapipes.timeseries.run_timeseries(
            self._gas_network, time_steps=range(1), verbose=False)

        self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # print(f'\u001b[33m{self._time_step:>3} -> reset\u001b[0m')

        return observation, info

    def step(self, action):
        """
        todo(lisca): Add documentation string.
        """

        # todo(lisca): remove this hack!
        action = action / 1000.0

        # Set the new mdot_kg_per_s of the storage controller.
        self._gas_network_controller._controller_storage.mdot_kg_per_s = action

        # ? stochastic (m_dot of the) sink

        # Simulate one hour.
        time_steps = range(self._time_step, self._time_step + 1)
        pandapipes.timeseries.run_timeseries(
            self._gas_network, time_steps=time_steps, verbose=False)

        observation = self._get_obs(action)
        reward      = self._get_reward(action)
        terminated  = False
        truncated   = False
        info        = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # if self._time_step % 1 == 0:
        #     print(f'\u001b[31m{self._time_step:>3} -> action: {action}\u001b[0m')
        #     print(f'\u001b[32m{self._time_step:>3} -> reward: {reward}\u001b[0m')

        self._action_previous = action
        self._time_step += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        todo(lisca): Add documentation string.
        """

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        todo(lisca): Add documentation string.
        """
        self._gas_network_visualizer.get_network_image(self._gas_network)

    def close(self):
        """
        todo(lisca): Add documentation string.
        """

        pass
