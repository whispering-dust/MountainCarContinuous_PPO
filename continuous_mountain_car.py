"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding


class Continuous_MountainCarEnvDense(gym.Env):
    """
    The agent (a car) is started at the bottom of a valley. For any given state
    the agent may choose to accelerate to the left, right or cease any
    acceleration. The code is originally based on [this code](http://incompleteideas.net/MountainCar/MountainCar1.cp)
    and the environment appeared first in Andrew Moore's PhD Thesis (1990):
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".

    ### Action

    The actual driving force is calculated by multiplying the power coef by power (0.0015)

    ### Reward

    Reward of 100 is awarded if the agent reached the flag (position = 0.45)
    on top of the mountain. Reward is decrease based on amount of energy consumed each step.

    ### Starting State

    The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The car position is more than 0.45. Episode length is greater than 200

    ### Arguments

    ```
    gym.make('MountainCarContinuous-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -9
        self.max_position = 6
        self.max_speed = 0.07
        self.goal_position = (
            5.6  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.screen = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action):

        position = self.state[0]
        last_position = position
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)
        slope = self._slope(position)
        angle = math.atan(slope)
        last_velocity = velocity
        velocity += force * self.power - 0.0025 * angle
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = -velocity
        
        # Convert a possible numpy bool to a Python bool.
        first_flag_position = -2.1
        #first_target = bool((position >= first_flag_position and velocity >= self.goal_velocity))
        final_target = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if last_position < first_flag_position and position >= first_flag_position:
            reward = 100
        if last_position >= first_flag_position and position < first_flag_position:
            reward = -100

        if final_target:
            reward = 200.0

        reward -= math.pow(action[0], 2) * 0.1 + (last_velocity ** 2 - velocity ** 2) * 10000 \
            + (self._height(last_position) - self._height(position)) * 50

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, final_target, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-6.6, high=-5.6), 0])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        y = 1.1* np.sin(0.8*xs - 2.8) + 0.16*xs+3
        
        # 修改后的函数
        return y
    
    def _slope(self, x):
        delta = 0.0001  # 微小变化量
        return (self._height(x + delta) - self._height(x - delta)) / (2 * delta)
    
    def render(self, mode="human"):
        screen_width = 1000
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10
        slope = self._slope(pos)
        angle = math.atan(slope)
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(angle)
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(angle)
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

class Continuous_MountainCarEnv(gym.Env):
    """
    The agent (a car) is started at the bottom of a valley. For any given state
    the agent may choose to accelerate to the left, right or cease any
    acceleration. The code is originally based on [this code](http://incompleteideas.net/MountainCar/MountainCar1.cp)
    and the environment appeared first in Andrew Moore's PhD Thesis (1990):
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation space is a 2-dim vector, where the 1st element represents the "car position" and the 2nd element represents the "car velocity".

    ### Action

    The actual driving force is calculated by multiplying the power coef by power (0.0015)

    ### Reward

    Reward of 100 is awarded if the agent reached the flag (position = 0.45)
    on top of the mountain. Reward is decrease based on amount of energy consumed each step.

    ### Starting State

    The position of the car is assigned a uniform random value in [-0.6 , -0.4]. The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The car position is more than 0.45. Episode length is greater than 200

    ### Arguments

    ```
    gym.make('MountainCarContinuous-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -9
        self.max_position = 6
        self.max_speed = 0.07
        self.goal_position = (
            5.6  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.screen = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action):

        position = self.state[0]
        last_position = position
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)
        slope = self._slope(position)
        angle = math.atan(slope)
        last_velocity = velocity
        velocity += force * self.power - 0.0025 * angle
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = -velocity
        
        # Convert a possible numpy bool to a Python bool.
        first_flag_position = -2.1
        #first_target = bool((position >= first_flag_position and velocity >= self.goal_velocity))
        final_target = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if last_position < first_flag_position and position >= first_flag_position:
            reward = 100
        if last_position >= first_flag_position and position < first_flag_position:
            reward = -100

        if final_target:
            reward = 200.0

        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, final_target, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-6.6, high=-5.6), 0])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        y = 1.1* np.sin(0.8*xs - 2.8) + 0.16*xs+3
        
        # 修改后的函数
        return y
    
    def _slope(self, x):
        delta = 0.0001  # 微小变化量
        return (self._height(x + delta) - self._height(x - delta)) / (2 * delta)
    
    def render(self, mode="human"):
        screen_width = 1000
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10
        slope = self._slope(pos)
        angle = math.atan(slope)
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(angle)
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))
        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(angle)
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False