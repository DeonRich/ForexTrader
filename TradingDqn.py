import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.tensor import Tensor
from typing import Tuple
from TradeEnv import TradingEnv
from collections import namedtuple, deque
from TradeRegression import DQN
from TradeRegression import calculateEMA
from test_env import EnvTest
from test_env import config as testConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
class config():
    model_output1       = "DQN_TrainingInfo/model_1.weights"
    model_output2       = "DQN_TrainingInfo/model_2.weights"
    sequence_len       = 100
    inChannels         = 9
    high               = 1.
    num_episodes_test  = 20
    log_freq           = 500
    eval_freq          = 20000
    saving_freq        = 50000

    clip_val           = 10
    nsteps_train       = 2000000
    batch_size         = 32
    buffer_size        = 500000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    lr                 = 0.00003
    lr_begin           = 0.00008
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/8
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = 750000
    learning_start     = 10000

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        s_batch = np.array(batch.state)
        a_batch = np.array(batch.action, dtype='uint8')
        r_batch = np.array(batch.reward, dtype='float32')
        sp_batch = np.array(batch.next_state)
        done_mask_batch = np.array(batch.done, dtype='bool')
        return s_batch, a_batch, r_batch, sp_batch, done_mask_batch

    def __len__(self):
        return len(self.memory)

class Q_Model(nn.Module):
    def __init__(self, n_actions, sequence_len, inChannels):
        super(Q_Model, self).__init__()
        self.compressed_len = 8
        flatten_len = self.compressed_len + int(sequence_len % self.compressed_len != 0)
        self.rnn = nn.GRU(inChannels, 128, batch_first=True, bidirectional=True)
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * flatten_len * 128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

    def forward(self, x):
        out, _    = self.rnn(x)
        out       = torch.flip(out,[1])
        lin_input = out[:,::out.size(1)//self.compressed_len]
        return self.lin(lin_input)

class testQModel(nn.Module):
    def __init__(self, n_actions, n_channels, img_height, img_width, state_history):
        super(testQModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels * state_history, 32, 8, stride = 4,  padding = (3 * img_height + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride = 2, padding = (img_height + 2) // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_height * img_width * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

    def forward(self, x):
        return self.net(x)

class DQN_nature(object):
    def __init__(self, env, config, logger=None):
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.env = env
        self.config = config
        self.steps = 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def run(self):
        self.initialize()
        self.train()

    def get_q_values(self, state, network='q_eval_network'):
        net = getattr(self, network)
        return net(state)

    def update_target(self):
        self.target_q1_network.load_state_dict(self.q1_network.state_dict())
        self.target_q2_network.load_state_dict(self.q2_network.state_dict())

    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(par for model in [self.q1_network, self.q2_network] for par in model.parameters())
    
    def initialize(self):
        self.initialize_models()
        self.q1_network.to(self.device)
        self.q2_network.to(self.device)
        self.target_q1_network.to(self.device)
        self.target_q2_network.to(self.device)
        self.add_optimizer()
        self.update_target()
    
    def initialize_models(self):
        self.q1_network = Q_Model(self.env.action_space.n, 
            self.config.sequence_len, 
            self.config.inChannels)
        self.q2_network = Q_Model(self.env.action_space.n, 
            self.config.sequence_len, 
            self.config.inChannels)
        self.target_q1_network = Q_Model(self.env.action_space.n, 
            self.config.sequence_len, 
            self.config.inChannels)
        self.target_q2_network = Q_Model(self.env.action_space.n, 
            self.config.sequence_len, 
            self.config.inChannels)
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        self.q1_network = testQModel(self.env.action_space.n, n_channels, img_height, img_width, self.config.state_history)
        self.q2_network = testQModel(self.env.action_space.n, n_channels, img_height, img_width, self.config.state_history)
        self.target_q1_network = testQModel(self.env.action_space.n, n_channels, img_height, img_width, self.config.state_history)
        self.target_q2_network = testQModel(self.env.action_space.n, n_channels, img_height, img_width, self.config.state_history)
        """
    def save(self):
        """
        Saves session
        """
        print("saving...")
        torch.save(self.q1_network.state_dict(), self.config.model_output1)
        torch.save(self.q2_network.state_dict(), self.config.model_output2)

    def get_action(self, state):
        eps_step = (self.config.eps_end - self.config.eps_begin) / self.config.eps_nsteps
        eps_threshold = self.config.eps_begin + eps_step * min(self.config.eps_nsteps, self.steps)
        if np.random.random() < eps_threshold:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            s = self.process_state(s)
            action_values = self.get_q_values(s, 'q1_network')
            action_values += self.get_q_values(s, 'q2_network')
            action_values = action_values.squeeze().to('cpu').tolist()
        action = np.argmax(action_values)
        return action, action_values
    
    def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        num_actions = self.env.action_space.n
        gamma = self.config.gamma

        Q_targ = rewards + torch.where(done_mask,  torch.zeros_like(rewards), gamma * torch.max(target_q_values,1)[0])
        Q_hat = torch.sum(F.one_hot(actions.long(), num_classes = num_actions) * q_values, axis=1)
        return F.mse_loss(Q_targ, Q_hat)

    def process_state(self, state : Tensor) -> Tensor:
        state = state.float()
        state /= self.config.high

        return state
    
    def train(self):
        replay_buffer = ReplayMemory(self.config.buffer_size)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]
        determ_eval = [self.evaluate(num_episodes=1, useEpsilon=False)]
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset().astype('uint8')
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                # chose action according to current Q and exploration
                best_action, q_vals = self.get_best_action(state)
                action                = self.get_action(state)

                #store q values
                max_q_values.append(max(q_vals))
                q_values += list(q_vals)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)
                new_state = new_state.astype('uint8')

                # store the transition
                replay_buffer.push(state, action, reward, new_state, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, self.config.lr)

                total_reward += reward
                
                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    if len(rewards) > 0:
                        self.update_averages(rewards, max_q_values, q_values, scores_eval)
                        msg =  f"{t + 1}, Loss: {loss_eval:.4f} - Avg_R, {self.avg_reward:.4f} - Max_R, {np.max(rewards):.4f} - "
                        msg += f"Grads {grad_eval:.4f} - Max_Q, {self.max_q:.4f}" 
                        print(msg)
                
                self.steps = t

                if done or t >= self.config.nsteps_train:
                    break

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                scores_eval += [self.evaluate()]
                determ_eval += [self.evaluate(num_episodes=1, useEpsilon=False)]

            rewards.append(total_reward)
        
        #final evaluations
        print("Training Done")
        self.save()
        scores_eval += [self.evaluate()]
        determ_eval += [self.evaluate(num_episodes=1, useEpsilon=False)]
        plt.clf()
        plt.xlabel("time")
        plt.ylabel("Average Reward")
        plt.plot(scores_eval)
        plt.savefig("EvaluationPlot.png")
        plt.clf()
        plt.xlabel("time")
        plt.ylabel("Average Reward")
        plt.plot(determ_eval)
        plt.savefig("BestActionPlot.png")

    def train_step(self, t, replay_buffer, lr):
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target()

        # occasionally save weights
        if t % self.config.saving_freq == 0:
            self.save()
            
        return loss_eval, grad_eval

    def update_step(self, t, replay_buffer, lr):
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)
        assert (self.q1_network is not None and self.target_q1_network is not None and
                self.q2_network is not None and self.target_q2_network is not None), \
            'WARNING: Networks not initialized. Check initialize_models'
        assert self.optimizer is not None, \
            'WARNING: Optimizer not initialized. Check add_optimizer'

        # Convert to Tensor and move to correct device
        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool, device=self.device)
        
        # Reset Optimizer
        self.optimizer.zero_grad()

        if np.random.random() < .5:
            network = 'q1_network'
        else:
            network = 'q2_network'
        # Run a forward pass
        s = self.process_state(s_batch)
        q_values = self.get_q_values(s, network)

        with torch.no_grad():
            sp = self.process_state(sp_batch)
            actions = torch.max(self.get_q_values(sp, network), 1)[1]
            target_q_values = self.get_q_values(sp, 'target_' + network)
            target_q_values = torch.sum(F.one_hot(
                                    actions.long(), 
                                    num_classes = self.env.action_space.n) * target_q_values, axis=1).unsqueeze(1)

        loss = self.calc_loss(q_values, target_q_values, 
            a_batch, r_batch, done_mask_batch)
        loss.backward()

        # Clip norm
        if network == 'q1_network':
            total_norm = torch.nn.utils.clip_grad_norm_(self.q1_network.parameters(), self.config.clip_val)
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(self.q2_network.parameters(), self.config.clip_val)

        # Update parameters with optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        return loss.item(), total_norm.item()

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = -21.
    
    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]
    
    def evaluate(self, env=None, num_episodes=None, useEpsilon=True):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            print("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayMemory(self.config.buffer_size)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset().astype("uint8")
            while True:

                if useEpsilon:
                    action = self.get_action(state)
                else:
                    action = self.get_best_action(state)[0]

                # perform action in env
                new_state, reward, done, info = env.step(action)
                new_state = new_state.astype('uint8')

                # store in replay memory
                replay_buffer.push(state, action, reward, new_state, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)     

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            print(msg)

        return avg_reward
    
    def evaluateModel(self):
        self.initialize()
        env = self.env
        state = env.reset().astype("uint8")
        self.q1_network.load_state_dict(torch.load(self.config.model_output1))
        self.q2_network.load_state_dict(torch.load(self.config.model_output2))
        rewardHistory = []
        total_reward = 0
        while True:

            action = self.get_best_action(state)[0]

            # perform action in env
            new_state, reward, done, info = env.step(action)
            new_state = new_state.astype('uint8')

            state = new_state

            # count reward
            total_reward += reward
            rewardHistory.append(total_reward)
            if done:
                break
        plt.clf()
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.plot(rewardHistory)
        plt.savefig("Evaluation.png")

def loadEnv():
    saveLocation = "TrainingInfo/"
    rawData = torch.load(saveLocation + "rawDataAndEma.pt").numpy()
    std, mean = torch.load(saveLocation + "std.pt"), torch.load(saveLocation + "mean.pt")
    encodings = torch.load(saveLocation + "rawEncodings.pt")
    std2, mean2 = np.std(rawData), np.mean(rawData)
    #use 7 and 8 for original test
    processed = (rawData - mean2) / std2
    processed = np.concatenate([processed[:, 1:] - processed[:, :-1], processed[:, 1:, 3:5]], axis=-1)
    env = TradingEnv(rawData, encodings, processed, config.sequence_len)
    return env

def loadEnvNumber(i):
    saveLocation = "TrainingInfo/"
    rawData = torch.load(saveLocation + "rawDataAndEma.pt").numpy()
    std, mean = torch.load(saveLocation + "std.pt"), torch.load(saveLocation + "mean.pt")
    encodings = torch.load(saveLocation + "rawEncodings.pt")
    std2, mean2 = np.std(rawData[7:8]), np.mean(rawData[7:8])
    #use 7 for original test
    processed = (rawData - mean2) / std2
    processed = np.concatenate([processed[i:i+1, 1:] - processed[i:i+1, :-1], processed[i:i+1, 1:, 3:5]], axis=-1)
    env = TradingEnv(rawData[i:i+1], encodings, processed, config.sequence_len)
    return env

def runSanityTest():
    env = EnvTest((8, 8, 6))
    model = DQN_nature(env, testConfig)
    model.run()

if __name__ == "__main__":
    start = time.time()
    env = loadEnv()
    # train model
    model = DQN_nature(env, config)
    model.run()
    model.evaluateModel()
    #runSanityTest()
    end = time.time()
    print(f'Completed in {(end-start)/3600:.4f} hrs')