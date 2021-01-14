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
    
    env.reset()
    obs = env.get_obs()
    state = env.get_state()
    env.close()
    
    if DISPLAY:
        LOAD_KEY, TRAIN_KEY = KEY[0]
        epsilon = 0.01
    else:
        LOAD_KEY, TRAIN_KEY = False, True
        epsilon = 0.6
    
    train_flag = False
    total_step = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型初始化
    model = QMIXMixing_net(args = (n_agents, len(state), [len(obs[i]) for i in range(n_agents)], [n_actions for i in range(n_agents)], device, LERNING_RATE)).to(device)
    if LOAD_KEY:
        print("Load model ...")
        check_point = torch.load("./param/Q_Mix_epoch_1600.pkl")
        model.load_state_dict(check_point)
    target_model = QMIXMixing_net(args = (n_agents, len(state), [len(obs[i]) for i in range(n_agents)], [n_actions for i in range(n_agents)], device, LERNING_RATE)).to(device)
    target_model.load_state_dict(model.state_dict())
    replay_buffer = Replaybuffer(args = 1000)

    # 训练开始
    for ep_i in range(N_EPISODES):
        epsilon = max(0.01, epsilon*DECAY_EP)
        env.reset()
        terminated = False
        obs = env.get_obs()
        state = env.get_state()

        # 记忆变量初始化
        episode_reward = 0. 
        episode_mem = []
        hidden_state = [torch.zeros(1, 1, 32).to(device) for _ in range(n_agents)]
        action_vec = [np.zeros(n_actions) for _ in range(n_agents)]
        actions = [0 for _ in range(n_agents)]

        while not terminated:
            actions_pre = copy.deepcopy(actions)
            action_pre_onehot_joint = copy.deepcopy(action_vec)
            actions = [0 for _ in range(n_agents)]
            action_mask_joint = []
            
            # 每个代理分开执行
            for agent_id in range(n_agents):
                action_mask = env.get_avail_agent_actions(agent_id)
                action, h_next = model.agent_model[0].get_action(torch.cat([torch.FloatTensor(obs[agent_id]).to(device), torch.FloatTensor(action_vec[agent_id]).to(device)]), hidden_state[agent_id], epsilon, action_mask)
                
                # update 记忆变量
                actions[agent_id] = action
                hidden_state[agent_id] = h_next
                action_vec[agent_id] = np.zeros(n_actions)
                action_vec[agent_id][action] = 1.
                action_mask = env.get_avail_agent_actions(agent_id)
                action_mask_joint.append(action_mask)
           
            # env step
            reward, terminated, _ = env.step(actions)
            loss_mask = 1. if not terminated else 0.
            done = 1. if terminated else 0.
            obs_next = env.get_obs()
            state_next = env.get_state()
            episode_mem.append((state, actions, action_vec, reward, state_next, done, obs, obs_next, actions_pre, action_pre_onehot_joint, action_mask_joint, loss_mask))
            
            if terminated:
                replay_buffer.save_trans(episode_mem)

            # 更新环境状态
            obs = obs_next
            state = state_next
            episode_reward += reward
            total_step += 1

            if TRAIN_KEY:
                # 权重覆盖
                if total_step % WEIGHT_COPY_INTERVAL == 0:
                    print("Weight copy!")
                    target_model.load_state_dict(model.state_dict())

        if TRAIN_KEY:
            if replay_buffer.mem_len >  BATCH_SIZE:
                QMIXtrain(replay_buffer, model, target_model, GAMMA, LERNING_RATE, BATCH_SIZE)
                train_flag = True 

            if (ep_i+1) % 200 == 0:
                print("Saving model param ... ")
                torch.save(model.state_dict(), PATH + "/Q_Mix_epoch_" + str(ep_i+1) + '.pkl')
                print("ok!")
        print("Total reward in episode {} = {:.3}, epsilon: {:.3},  training:  {}".format(ep_i, episode_reward, epsilon, train_flag))


    env.close()





if __name__ == "__main__":
     main()