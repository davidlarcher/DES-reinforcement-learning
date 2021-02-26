import argparse
import os

import ray
from ray import tune

from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from elevator import create_env
from elevator_env import ElevatorEnv
from agent_configs import config_PPO, config_SAC, config_DDPG
from pprint import pprint

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--stop-reward", type=float, default=1000.)
parser.add_argument("--num-wt-rows", type=int, default=1)
parser.add_argument("--num-wt-cols", type=int, default=2)

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_gpus=4)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    env_config = {
        "env_config": {
            "num_lifts": 3,
            "lift_capacity": 4,
            "num_floors": 16,
            "max_queue": 10,
            "max_mean_waiting_time": 500,
        }}

    general_config = {
        "env": ElevatorEnv,
        "model": {
            "custom_model": "my_model",
        },
        "framework": "torch" if args.torch else "tf",
        "callbacks": DefaultCallbacks,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    agent = {}
    agent_config = {}
    if args.run == "PPO":
        agent_config = config_PPO
    elif args.run == "SAC":
        agent_config = config_SAC
    elif args.run == "DDPG":
        agent_config = config_DDPG

    config = {
        **env_config,
        **agent_config,
        **general_config
    }
    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # restore="/home/david/ray_results/SAC/SAC_FarmEnv_5aa8e_00000_0_2021-01-21_18-23-19/checkpoint_199/checkpoint-199",
    )
    ray.shutdown()
