from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

import numpy as np
from collections import deque

from typing import List
from tqdm import tqdm

import sys 
sys.path.append("/Users/emrebaloglu/Documents/RL/nlp-gym")
from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label.env import MultiLabelEnv
from nlp_gym.envs.multi_label.reward import F1RewardFunction
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from rich import print

class Softmax_Policy_Dense_Layers(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_layer_dims: List[int]):
        """
        Initialize a policy with softmax probabilities, estimated with a fully-connected neural network.

        Args:
            state_size (int): Size of the states.
            action_size (int): Size of the actions.
            hidden_layer_dims (List[int]): List of units in hidden layers. 
        """
        super(Softmax_Policy_Dense_Layers, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}.")
        self.hidden_layers = []
        if len(hidden_layer_dims) > 0:
            self.fc1 = nn.Linear(state_size, hidden_layer_dims[0]).to(self.device)
            for ii in range(len(hidden_layer_dims) - 1):
                hidden_layer = nn.Linear(hidden_layer_dims[ii], hidden_layer_dims[ii+1]).to(self.device)
                self.hidden_layers.append(hidden_layer)
            out_layer = nn.Linear(hidden_layer_dims[-1], action_size).to(self.device)
            self.hidden_layers.append(out_layer)
        else: 
            self.fc1 = nn.Linear(state_size, action_size).to(self.device)

        # self = self.to(self.device)
        
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        if len(self.hidden_layers) > 0:
            x = F.relu(self.fc1(x))
            for layer_ix in range(len(self.hidden_layers)-1):
                x = F.relu(self.hidden_layers[layer_ix](x))
            
            x = F.softmax(self.hidden_layers[-1](x), dim=-1)
        else:
            x = F.softmax(self.fc1(x), dim=-1)
        
        return x
    
    def predict(self, x):
        return self.forward(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state)#.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce_algorithm(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in tqdm(range(1, n_training_episodes+1), desc=f"Training the agent with REINFORCE..."):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        ## Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## We calculate the return by sum(gamma[t] * reward[t]) 
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        avg_score = np.mean(scores_deque)
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
        
    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0
    
        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward
            
            if done:
                break
            state = new_state
            
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def eval_model(model, env):
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    while not done:
        action = np.argmax(model.predict(obs).detach().numpy())
        obs, rewards, done, info = env.step(action)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Text: {env.current_sample.text}")
    print(f"Predicted Label {actions}")
    print(f"Oracle Label: {env.current_sample.label}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


if __name__ == "__main__":
    # data pool
    pool = ReutersDataPool.prepare(split="train")
    labels = pool.labels()

    # reward function
    reward_fn = F1RewardFunction()

    # multi label env
    env = MultiLabelEnv(possible_labels=labels, max_steps=10, reward_function=reward_fn,
                        return_obs_as_vector=True)
    for sample, weight in pool:
        env.add_sample(sample, weight)

    # check the environment
    check_env(env, warn=True)

    state_dim = len(env.reset())
    action_dim = len(pool.labels())+1
    print(state_dim, action_dim)

    hyperparameters = {
        "n_training_episodes": int(1e+5),
        "n_evaluation_episodes": 1,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-3,
        "env_id": None,
        "state_space": state_dim,
        "action_space": action_dim,
    }   

    policy = Softmax_Policy_Dense_Layers(state_dim, action_dim, [128, 64, 64, 64])

    optimizer = Adam(policy.parameters(), lr=hyperparameters["lr"])

    for ii in range(10):
        scores = reinforce_algorithm(env, policy,
                    optimizer,
                    hyperparameters["n_training_episodes"], 
                    hyperparameters["max_t"],
                    hyperparameters["gamma"], 
                    1000)
    
        eval_model(policy, env)
