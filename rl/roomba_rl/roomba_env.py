import gym
from gym.error import DependencyNotInstalled
from gym import spaces
import numpy as np
from roomba import *
from particle import Pose, ParticleMap
from sensor import Sensor
import math

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )
FPS=10
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
P_SCALE = 10.0

VIEWPORT_W = 600
VIEWPORT_H = 400

LINEAR_SPEED=20
ROTATIONAL_SPEED=1.0

N_PARTICLES=100
PARTICLE_SPEED=2

SENSOR_DETECTION_THRESHOLD=50

COLLISION_DIST=8

def rotate(dp,theta):
    dx, dy = dp
    return (
        dx*math.cos(theta) - dy*math.sin(theta),
        dx*math.sin(theta) + dy*math.cos(theta)
    )

class RoombaEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def _init_states(self):
        roomba_start_x = VIEWPORT_W // 2
        roomba_start_y = VIEWPORT_H // 2
        self._roomba = Roomba(
            pos=Pose(
                x=roomba_start_x,
                y=roomba_start_y,
                theta=0.
            ), 
            dx=LINEAR_SPEED/FPS,
            dtheta=ROTATIONAL_SPEED/FPS
        )
        self._particles = ParticleMap(
            N_PARTICLES,
            VIEWPORT_W,
            VIEWPORT_H, 
            PARTICLE_SPEED,
            collision_dist=COLLISION_DIST
        )
        self._sensor = Sensor(SENSOR_DETECTION_THRESHOLD)
        self._i = 0


    def __init__(self, render_mode="human", max_episode_steps=1000) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(4)
        low = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        high = np.array([1.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self._init_states()
        self._max_episode_steps = 200
        self.render_mode = render_mode
        self.screen: pygame.Surface = None
        self.clock = None
        self.terminated = False
        
        
    def step(self, action):
        # TODO: clean this up
        if self._i >= self._max_episode_steps:
            self.terminated = True
        if self.terminated:
            sensor_output = self._sensor.sense(self._roomba, self._particles)
            return np.array(sensor_output, dtype=np.float32), 0, self.terminated, {}
        self._roomba.move(action)
        self._particles.move()
        reward = 0
        if self._particles.detect_collision(self._roomba.pose):
            reward = -100
            self.terminated = True
        # If moving forward, gain point
        elif action == 0:
            reward = 1
        # If moving backwards, then penalize
        elif action == 2:
            reward = -1
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        self._i += 1
        return np.array(sensor_output, dtype=np.float32), reward, self.terminated, {}

    # Need for the initial state
    def reset(self):
        self._init_states()
        self.terminated = False
        sensor_output = self._sensor.sense(self._roomba, self._particles)
        return np.array(sensor_output, dtype=np.float32)

        

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        # Draw roomba
        roomba_pose = self._roomba.pose
        x = int(roomba_pose.x) 
        y = int(roomba_pose.y) 
        dps = [
            (2, 0),
            (6, 6),
            (0, 8),
            (-6, 6),
            (-8, 0),
            (-6, -6),
            (0, -8),
            (6, -6)
        ]
        rotated_dp = [rotate(p, roomba_pose.theta) for p in dps]
        points = [
            (int(x + p[0]), int(y + p[1])) for p in rotated_dp
        ]
        pygame.draw.polygon(
            self.surf,
            color=(255, 255, 255),
            points=points
        )

        # Draw objects
        for pos in self._particles.particles:
            if pos.x < 0 or pos.y < 0:
                continue
            if pos.x > VIEWPORT_W or pos.y > VIEWPORT_H:
                continue
            pygame.draw.circle(
                self.surf,
                color=(255,0,0),
                center=(int(pos.x), int(pos.y)),
                radius=2,
            )

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

if __name__ == '__main__':
    env = RoombaEnv()
    for i in range(100):
        env.render()
        state, reward, terminated, _ = env.step(env.action_space.sample())
        print(state)
        print(reward)
        print("===========")
        if terminated:
            exit(0)
