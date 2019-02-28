import numpy as np
import random
import gym
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    '''
    A class that represents the policy network, takes in a 
    state and produces the action-values of taking each
    valid action in that state.

    '''
    def __init__(self, state_space, action_space, hidden_size):
        '''
        Arguments:
            state_space: the StateSpace object of the environment
            action_space: the ActionSpace object of the environment
            hidden_size: the number of neurons in the hidden layer
        '''
        super().__init__()

        self.state_space = state_space
        self.action_space = action_space

        # Extract the state space and action space dimensions.
        input_dim = state_space.high.shape[0]
        output_dim = action_space.n

        # Define the architecture of the neural network.
        self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            )

    def forward(self, state):
        '''
        Calculates the forward pass of the neural network.

        Arguments: 
            state: the state of the environments

        Returns: 
            output: the corresponding action-values from the network.
        '''
        output = self.fc(state)

        return output

class QAgent(object):
    '''
    A class representing an agent that follows the Q-learning 
    algorithm for learning an optimal policy.
    '''
    def __init__(self, policy_network, target_network, epsilon):
        '''
        Arguments:
            policy_network: the network that is updated at each step and 
                            which the current action-value of a state is computed.
            target_network: the network that is used to construct the target action-values
                            and is synchronised periodically.
            epsilon: determines the amount of exploration in e-greedy.
        '''
        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon = epsilon

        # Instantiate memory buffer.
        self.memory = []
        
    def select_action(self, state, action_space):
        '''
        Given a state, chooses an action in the action space
        according to e-greedy of the action-value function.

        Arguments:
            state: the state of the environment
            action_space: the action space of the environment.
        '''

        # Generate a random uniform between 0 and 1
        rand = np.random.rand()
    
        # If rand is less than eps, choose random action in action space.
        if rand < self.epsilon:
            action = np.random.randint(action_space.n)

        # Else, choose action greedily according to action-value function.
        else:
            output = self.policy_network.forward(state)
            action = torch.argmax(output)

        action = torch.tensor(action).view(-1, 1)

        return action


if __name__ == '__main__':
    # Set up the CartPole Environment.
    env = gym.make("MountainCar-v0").env
        
    # Retrieve the state space and action space objects for CartPole.
    state_space = env.observation_space
    action_space = env.action_space

    # Specify hyper-parameters for training.
    batch_size = 32
    hidden_size = 32
    memory_flush = 2000

    epsilon = 0.01
    discount = 0.99
    episodes = 1000
    # Set up the policy network (the one to be updated at every step)
    policy_network = PolicyNetwork(state_space, action_space, hidden_size).double()
    
    # Set up the target network (used to calculate the target)
    target_network = PolicyNetwork(state_space, action_space, hidden_size).double()

    # Instantiate the Q-learning agent
    agent = QAgent(policy_network, target_network, epsilon=epsilon)

    # Set up the loss function and optimiser for the NN.
    criterion = nn.MSELoss()
    optimiser = torch.optim.RMSprop(policy_network.parameters())
    
    # Store the end reward of each episode
    episode_rewards = []
    
    # For each episode
    for episode in range(episodes):
        # Instantiate the environment
        state = torch.tensor(env.reset())
        score = 0
        loss = 0
        
        # Every ten episodes, synchronise the two networks
        if episodes % 10 == 0:
            target_network.load_state_dict(policy_network.state_dict())

        # Boolean to tell us whether the episode is finished.
        done = False
        while not done:
            env.render()
            
            # Zero-grad the optimiser
            optimiser.zero_grad()
            
            # Given a state, let the agent select an action
            action = agent.select_action(state, action_space)
            
            # Retrieve the next state and reward from the environment
            next_state, reward, done, info = env.step(action.data.numpy()[0, 0])
            next_state = torch.tensor(next_state).type(torch.DoubleTensor)

            reward = torch.tensor(reward).type(torch.FloatTensor).view(-1, 1)

            # Update accumulative score
            score += reward

            # Store the experience into the agent's experience buffer.
            agent.memory.append((state, action, reward, next_state, done))

            # Forget the oldest experience if agent stores over memory_flush experience.
            if len(agent.memory) > memory_flush:
                del agent.memory[0]

            # If the agent's buffer is greater than the batch_size
            if len(agent.memory) > batch_size:
                
                # Sample a batch of experience from the buffer
                minibatch = random.sample(agent.memory, batch_size)
                
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*minibatch)

                batch_state = torch.stack(batch_state)
                batch_action = torch.stack(batch_action).view(-1, 1)
                batch_reward = torch.stack(batch_reward).view(-1, 1)
                batch_next_state = torch.stack(batch_next_state)

                # Calculate the max_a' [Q(s', a')] using the target network
                Q_next_max, _ = torch.max(target_network.forward(batch_next_state).detach(), dim=1)
                Q_next_max = Q_next_max.view(-1, 1).detach()

                # Discount the action-values
                discounted_Q = (discount * Q_next_max).type(torch.FloatTensor)
                batch_reward = batch_reward.type(torch.FloatTensor)

                # Calculate Q(s) using the policy network
                Q_current = policy_network.forward(batch_state)
                
                # Create the target network
                Q_target = Q_current.clone()

                for i in range(len(Q_target)):
                    if not batch_done[i]:
                        Q_target[i, batch_action[i][0]] = batch_reward[i] + discounted_Q[i]

                    else:
                        Q_target[i, batch_action[i][0]] = batch_reward[i]

                # Calculate the loss between the current and target action-values.
                loss = criterion(Q_current, Q_target)

                # Retrieve gradients of the network
                loss.backward()
                
                # Take a gradient descent step.
                optimiser.step()

            # Set the current state the next state.
            state = next_state

            # Print progress.
            if done:
                print("Episode %d -> Loss: %.4f\t Reward: %d \t(eps: %.4f)" % (episode, loss, score, agent.epsilon))
                break

        # Store the total rewards of each episode
        episode_rewards.append(score.data.numpy()[0, 0])