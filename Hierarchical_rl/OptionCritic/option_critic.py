import torch
from torch import nn, optim
import numpy as np
import random
import gym
import torch.nn.functional as F

class Option_critic(nn.Module):
    def __init__(self, input_size, output_size, option_num):
        super(Option_critic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.option_num = option_num
        self.option_actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.terminal_net = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.option_policy = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.option_num)
        )
        self.option_net = nn.ModuleList(self.option_actor for _ in range(self.option_num))

        self.actor_optim = optim.Adam([{'params': self.option_net.parameters()},{'params': self.option_policy.parameters()}], lr = 1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = 1e-3)
        self.terminal_optim = optim.Adam(self.terminal_net.parameters(), lr = 1e-3)
        self.trans_buffer =  []


    def save_trans(self, trans):
        self.trans_buffer.append(trans)

    def get_option(self, inputs):
        option_prob = self.option_policy(inputs)
        return torch.argmax(option_prob, -1).item()
    
    def get_action(self, option_id, inputs):
        action_prob = self.option_net[option_id](inputs)
        prob = F.softmax(action_prob, -1)
        return torch.distributions.Categorical(prob).sample().item()
    
    def cal_critic(self, inputs):
        critic_val = self.critic(inputs)
        return critic_val
    
    def is_terminal(self, inputs):
        terminal_p = self.terminal_net(inputs).item()
        if terminal_p > 0:
            return True
        else:
            return False

    
    def train(self, gamma):
        for trans in self.trans_buffer:
            s, a, r, s_next, done, option_id = trans
            q_val = self.critic(torch.FloatTensor(s))
            q_target = torch.FloatTensor([r]) + gamma * self.critic(torch.FloatTensor(s_next)) * torch.FloatTensor([done])
            advantage = q_target.detach() - q_val
            terminal_op = self.terminal_net(torch.FloatTensor(s))
            policy_op = F.softmax(self.option_net[option_id](torch.FloatTensor(s)), -1)[a]
            
            critic_loss = advantage ** 2
            self.critic_optim.zero_grad()
            critic_loss.backward(retain_graph = True)
            self.critic_optim.step()

            actor_loss = -torch.log(policy_op) * advantage.detach()
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph = True)
            self.actor_optim.step()

            terminal_loss = -torch.log(terminal_op) * advantage.detach()
            self.terminal_optim.zero_grad()
            terminal_loss.backward(retain_graph = True)
            self.terminal_optim.step()
        self.trans_buffer = []


if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    obversation = env.reset()

    print("Obversation space:",env.observation_space)
    print("Action space:",env.action_space.n)

    # 超参数设置
    gamma = 0.98
    learning_rate = 0.001
    epoch_num = 10000   # 回合数
    max_steps = 400   # 最大步数
    train_flag = False

    # 初始化
    model = Option_critic(4, 2, 3)
    score_list = []
    loss_list = []

    for i in range(epoch_num):
        epsilon = max(0.01,0.4-0.01*(i)/200)
        s = env.reset()
        score = 0.
        option_choice = model.get_option(torch.FloatTensor(s))
        for j in range(max_steps):
            # env.render()
            if model.is_terminal(torch.FloatTensor(s)):
                option_choice = model.get_option(torch.FloatTensor(s))
            a = model.get_action(option_choice, torch.FloatTensor(s))
            s_next, r, done, info = env.step(a)
            done_flag = 0.0 if done else 1.0
            model.save_trans((s, a, r, s_next, done_flag, option_choice))
            score += r
            s = s_next
            if done:
                train_flag = True
                model.train(gamma)
                break
        score_list.append(score)
        print("{} epoch score: {}  training: {}".format(i+1,score,train_flag))
    plot_curse(score_list,loss_list)
    env.close()

