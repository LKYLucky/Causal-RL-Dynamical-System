import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from env import LotkaVolterraEnv
import torch
#import torch.nn.functional as F
#from stable_baselines.sac.policies import MlpPolicy
#from stable_baselines import SAC

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)
        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        return action


class Actor(torch.nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)
        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        return action


class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.net(state)

def convert_to_vec(action):
    if action == 0:
        u = np.array([0,0])
    elif action == 1:
        u = np.array([0, 1])
    elif action == 2:
        u = np.array([1, 0])

    return u

def discounted_rewards(rewards, gamma):
    returns = np.zeros_like(rewards)
    J = 0.0
    for t in reversed(range(len(rewards))):
        J = J * gamma + rewards[t]
        returns[t] = J
    return returns

def update_policy_reinforce(states, actions, returns,model, optimizer):
    for s in range(len(states)):
        state, action, J = states[s], actions[s], returns[s]
        probs = model(state)  # https://pytorch.org/docs/stable/distributions.html
        m = torch.distributions.Categorical(probs=probs)
        log_prob = m.log_prob(action)
        loss = - log_prob * J

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer,critic_optimizer):
    for s in range(len(states)):
        state, action, R = states[s], actions[s], returns[s]
        probs = actor(state)  # https://pytorch.org/docs/stable/distributions.html
        policy = torch.distributions.Categorical(probs=probs)
        value = torch.IntTensor.item(critic(state))

        advantage = R - value
        actor_optimizer.zero_grad()
        actor_loss = - (policy.log_prob(action) * advantage)
        actor_loss.backward()
        actor_optimizer.step()

        # loss for critic (MSE)
        critic_optimizer.zero_grad()
        critic_loss = torch.tensor(advantage, requires_grad=True).pow(2)
        critic_loss.backward()
        critic_optimizer.step()

def run(Z_history, algorithm):
    model = Model()
    actor = Actor()
    critic = Critic()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    print("obs space",env.observation_space)
    print("action_space",env.action_space.n)

    #train
    max_episode = 100
    n_episode = 0
    max_step = 100
    scores = []
    prob_list = []

    while n_episode < max_episode:
        #done = False
        states = []
        actions = []
        rewards = []
        observation = torch.tensor(env.reset(), dtype=torch.float)
        for i in range(max_step):#while not done:
            if algorithm == "reinforce":
                action = model.select_action(observation)
            elif algorithm == "a2c":
                action = actor.select_action(observation)
<<<<<<< HEAD
=======
            '''
            if action == 0:
                zeros += 1
            elif action == 1:
                ones += 1
            elif action == 2:
                twos += 1
            '''
>>>>>>> 5a2c576a0015b6c2d1c6e7bde4936af9efb2c6cc

            #convert action to vectors (0,0),(0,1),(1,0)
            u = convert_to_vec(action)
            obs, reward, done, info, Z= env.step(u)
            Z_history = np.concatenate((Z_history, Z), 0)
            states.append(observation)

            actions.append(torch.tensor(action, dtype=torch.int))

            rewards.append(reward)
            observation = torch.tensor(obs, dtype=torch.float)
<<<<<<< HEAD

        x = torch.tensor([2, 1], dtype=torch.float)
=======
        '''
        zero_prob = zeros/max_step
        one_prob = ones/max_step
        two_prob = twos/max_step
        '''
        x = torch.tensor([1, 1], dtype=torch.float)
>>>>>>> 5a2c576a0015b6c2d1c6e7bde4936af9efb2c6cc
        y = model.forward(x)
        y = y.tolist()
        prob_list.append(y)
        scores.append(sum(rewards))
        #prob_list.append([zero_prob, one_prob, two_prob])
        n_episode += 1

        returns = discounted_rewards(rewards, gamma=0.95)
        if algorithm == "reinforce":
            update_policy_reinforce(states, actions, returns, model, optimizer)
        elif algorithm == "a2c":
            update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer,critic_optimizer)

    #eval
    #done = False
    observation = torch.tensor(env.reset(), dtype=torch.float)
    #env.render()

    rewards = []
    total_rewards = 0
    for i in range(max_step):#while not done:

        if algorithm == "reinforce":
            action = model.select_action(observation)
        elif algorithm == "a2c":
            action = actor.select_action(observation)

        u = convert_to_vec(action)
        obs, reward, done, info,Z = env.step(u)
        observation = torch.tensor(obs, dtype=torch.float)
        rewards.append(reward)
        total_rewards += reward
        #env.render()

    print('reward', total_rewards)
    print("p_list",prob_list)
<<<<<<< HEAD

=======
    '''
>>>>>>> 5a2c576a0015b6c2d1c6e7bde4936af9efb2c6cc
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()

    '''
<<<<<<< HEAD
=======
    '''
>>>>>>> 5a2c576a0015b6c2d1c6e7bde4936af9efb2c6cc
    tt = np.linspace(0, max_step, Z_history.shape[0])
    plt.plot(tt, Z_history[:, 0], 'kx', label='Z0')
    plt.plot(tt, Z_history[:, 1], 'rx', label='Z1')
    plt.legend(shadow=True, loc='lower right')
    plt.xlabel('t')
    plt.ylabel('n')
    plt.show()
    #plt.savefig('./lv.png')
<<<<<<< HEAD


=======
>>>>>>> 5a2c576a0015b6c2d1c6e7bde4936af9efb2c6cc
    '''


    plt.figure()
    tt = np.linspace(0, max_episode, max_episode)

    zeros_prob = [item[0] for item in prob_list]
    ones_prob = [item[1] for item in prob_list]
    twos_prob = [item[2] for item in prob_list]
    plt.plot(tt, zeros_prob, 'kx', label='Item 0')
    plt.plot(tt, ones_prob, 'rx', label='Item 1')
    plt.plot(tt, twos_prob, 'o', label='Item 2')
    plt.legend(shadow=True, loc='lower right')
    plt.xlabel('Episode #')
    plt.ylabel('Probabilities')
    plt.show()
    #plt.savefig('./lv.png')
    
    exit()
    env.close()

N = 2 # number of species
d = N + N*N # [ALTERNATE:] int(N + N*(N-1)/2) # we assume shared rate constants for Z_i*Z_j in d(Z_i)/dt and d(Z_j)/dt
tau = 1
dt = 1e-3
Z_current = np.array([1.0, 1.5]) # np.ones((N,))
env = LotkaVolterraEnv(N, tau, dt, Z_current)
Z_history = np.expand_dims(Z_current, 0)
env.state = Z_current

u = np.array([0,0.1])

theta_hat = np.ones(d)

run(Z_history, algorithm = "reinforce")
#run(Z_history, algorithm = "a2c")

