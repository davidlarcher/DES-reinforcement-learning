import gym
from gym.spaces import Discrete, MultiDiscrete, Box
import numpy as np
import salabim as sim
from elevator import create_env


class ElevatorEnv(gym.Env):

    def __init__(self, config):
        self.num_lifts = config["num_lifts"]
        self.lift_capacity = config["lift_capacity"]
        self.num_floors = config["num_floors"]
        self.max_queue = config["max_queue"]
        self.max_mean_waiting_time = config["max_mean_waiting_time"]
        self.building = {}
        self.observation = []
        self.count_steps = 0

        # 3 actions per lift (down, still, up)
        self.action_space = MultiDiscrete(np.full((self.num_lifts,), 3, dtype=np.int64))

        # define observation space

        lift_obs = np.array([])
        for i in range(self.num_lifts):
            obs_for_one_lift = np.array([self.num_floors, 1, self.lift_capacity], dtype=np.int64)
            lift_obs = np.concatenate((lift_obs, obs_for_one_lift), axis=0)

        floor_obs = []
        for i in range(self.num_floors):
            obs_for_one_floor = np.array([self.max_queue, self.max_queue], dtype=np.int64)
            floor_obs = np.concatenate((floor_obs, obs_for_one_floor), axis=0)

        observation_high = np.concatenate((lift_obs, floor_obs), axis=0)
        observation_low = np.zeros_like(observation_high)

        self.observation_space = Box(low=observation_low, high=observation_high, dtype=np.uint8)

    def reset(self):
        sim.reset()
        self.building = create_env()
        self.count_steps = 0
        # Get the Observations from the simulation
        self.observation = self.get_observation()
        return self.observation

    def get_observation(self):
        observation = np.array([], dtype=np.int64)
        for lift in self.building.lifts:
            observation = np.concatenate((observation, np.array([lift.floor.n])), axis=0) # current floor
            observation = np.concatenate((observation, np.array([0 if lift.dooropen else 1])), axis=0) # door open or not
            observation = np.concatenate((observation, np.array([len(lift.visitors)])), axis=0) # number of PAX
        for floor in self.building.floors:
            observation = np.concatenate((observation, np.array([min(floor.count_in_direction(+1), self.max_queue)])), axis=0) # visitors willing to go up at this particula floor
            observation = np.concatenate((observation, np.array([min(floor.count_in_direction(-1), self.max_queue)])), axis=0) # visitors willing to go down at this particula floor
        return observation

    def step(self, action):

        # check actions validity
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        #  Execute the actions
        # print(f'actions are {action}')
        for (i, lift) in enumerate(self.building.lifts):
            lift.direction = action[i] - 1
        self.building.run(1)

        # Get the Observations from the simulation
        self.observation = self.get_observation()

        # check observation
        err_msg = "%r (%s) invalid" % (self.observation, type(self.observation))
        assert self.observation_space.contains(self.observation), err_msg



        # reward calc

        mean_waiting_time = 0
        for floor in self.building.floors:
            mean_waiting_time += floor.visitors.length_of_stay.mean()
        mean_waiting_time /= self.num_floors
        print(f'mean waiting time       {mean_waiting_time}')
        reward = 1 / (mean_waiting_time + 0.1) / 10

        self.count_steps += 1

        done = False
        # Done Evaluation
        if self.count_steps == 500:
            done = True
        return self.observation, reward, done, {}


