from smac.env import StarCraft2Env
import numpy as np
import torch
import random
from torch import nn, optim
import torch.functional as F
from model import *
import copy



def main():
    env = StarCraft2Env(map_name="3m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    # 获取环境信息
    env.reset()
    obs = env.get_obs()
    state = env.get_state()
    env.close()
    # print(n_actions, n_agents)

    # Hyperparameters
    LERNING_RATE = 8e-4
    GAMMA = 0.98
    N_EPISODES = 10000
    TRAIN_BEGIN = 10000
    BATCH_SIZE = 32
    WEIGHT_COPY_INTERVAL = 5000
    DECAY_EP = 0.999
    PATH = './param'
    DISPLAY = False
    KEY = [[True, False], [False, True]]
    if DISPLAY:
        LOAD_KEY, TRAIN_KEY = KEY[0]
        LERNING_RATE = 5e-5
        epsilon = 0.01
    else:
        LOAD_KEY, TRAIN_KEY = KEY[1]
        LERNING_RATE = 9e-4
        epsilon = 0.4
    
    train_flag = False
    total_step = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型初始化
    model = Mixing_net(args = (n_agents, len(state), [len(obs[i]) for i in range(n_agents)], [n_actions for i in range(n_agents)], device, LERNING_RATE)).to(device)
    if LOAD_KEY:
        print("Load model ...")
        check_point = torch.load("./param/Q_Mix_epoch_1600.pkl")
        model.load_state_dict(check_point)
    target_model = Mixing_net(args = (n_agents, len(state), [len(obs[i]) for i in range(n_agents)], [n_actions for i in range(n_agents)], device, LERNING_RATE)).to(device)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = Replaybuffer(args = 1000)

    
    # train
    for e in range(N_EPISODES):
        epsilon = max(0.01, epsilon*DECAY_EP)
        env.reset()
        episode_mem = []
        terminated = False
        episode_reward = 0
        obs = env.get_obs()
        state = env.get_state()
        hidden_state = [torch.zeros(1, 1, 32).to(device) for _ in range(n_agents)]
        action_vec = [torch.zeros(n_actions).to(device) for _ in range(n_agents)]
        actions = [0 for i in range(n_agents)]
        while not terminated:
            action_pre = copy.deepcopy(actions)
            actions = [0 for i in range(n_agents)]
            action_mask_ep = []
            for agent_id in range(n_agents):
                action_mask = env.get_avail_agent_actions(agent_id)
                action, h_next = model.agent_model[0].get_action(torch.cat([torch.FloatTensor(obs[agent_id]).to(device), action_vec[agent_id]]), hidden_state[agent_id], epsilon, action_mask)
                actions[agent_id] = action
                hidden_state[agent_id] = h_next
                action_vec[agent_id] = torch.zeros(n_actions).to(device)
                action_vec[agent_id][action] = 1.
                action_mask = env.get_avail_agent_actions(agent_id)
                action_mask_ep.append(action_mask)
            reward, terminated, _ = env.step(actions)
            obs_next = env.get_obs()
            state_next = env.get_state()
            episode_mem.append((state, actions, reward, state_next, terminated, obs, obs_next, action_pre, action_mask_ep))
            if terminated:
                replay_buffer.save_trans(episode_mem)
            
            # 更新环境状态
            obs = obs_next
            state = state_next
            episode_reward += reward
            total_step += 1
            
            # train begin
            if TRAIN_KEY:
                # 权重覆盖
                if total_step % WEIGHT_COPY_INTERVAL == 0:
                    print("Weight copy!")
                    target_model.load_state_dict(model.state_dict())

        if TRAIN_KEY:
            if replay_buffer.mem_size >  BATCH_SIZE:
                train(replay_buffer, model, target_model, GAMMA, LERNING_RATE, BATCH_SIZE)
                train_flag = True 

            if (e+1) % 200 == 0:
                print("Saving model param ... ")
                torch.save(model.state_dict(), PATH + "/Q_Mix_epoch_" + str(e+1) + '.pkl')
                print("ok!")
        print("Total reward in episode {} = {}, epsilon: {},  training:  {}".format(e, episode_reward, epsilon, train_flag))


    env.close()

if __name__ == "__main__":
    main()