import torch 
import random
import numpy as np
import copy

from smac.env import StarCraft2Env
from param import *
from train import *
from model import *

def main():
    env = StarCraft2Env(map_name = SCEN_NAME)
    env_info = env.get_env_info()

    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']
    print(n_actions)


if __name__ == "__main__":
     main()