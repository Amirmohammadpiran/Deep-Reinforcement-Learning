import os
import gc
import torch
import pygame
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        
        self.capacity = capacity
        
        self.states       = deque(maxlen=capacity)
        self.actions      = deque(maxlen=capacity)
        self.next_states  = deque(maxlen=capacity)
        self.rewards      = deque(maxlen=capacity)
        self.dones        = deque(maxlen=capacity)
        
        
    def store(self, state, action, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """
        
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        
        
    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """
        
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones
    
    
    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque 
        represents the length of the entire memory.
        """
        
        return len(self.dones)
    
    
class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consists of Fully Connected (FC) layers with ReLU activation functions.
    """
    
    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.
        
        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """
        
        super(DQN_Network, self).__init__()
                                                          
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
            )
        
        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
        
    def forward(self, x):
        """
        Forward pass of the network to find the Q-values of the actions.
        
        Parameters:
            x (torch.Tensor): Input tensor representing the state.
        
        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """
        
        Q = self.FC(x)    
        return Q
    
        
class DQN_Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the boltzman policy.
    """
    
    def __init__(self, env,  
                  clip_grad_norm, learning_rate, discount, memory_capacity):
        
        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
                     
        # RL hyperparameters
        self.discount      = discount

        self.action_space  = env.action_space
        self.action_space.seed(seed) # Set the seed to get reproducible results when sampling the action space 
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)
        
        # Initiate the network models
        self.main_network = DQN_Network(num_actions=2, input_dim=4).to(device)
        self.target_network = DQN_Network(num_actions=2, input_dim=4).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
                

    def select_action(self, state, temperature):
        """
        Selects an action using boltzman strategy OR based on the Q-values.
        
        Parameters:
            state (torch.Tensor): Input tensor representing the state.
        
        Returns:
            action (int): The selected action.
        """
        with torch.no_grad():
            Q_values = self.main_network(state)
            probabilities = torch.softmax(Q_values / temperature, dim=0)
            action = torch.multinomial(probabilities, 1).item()
    
        return action
   

    def learn(self, batch_size, done):
        
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        actions       = actions.unsqueeze(1)
        rewards       = rewards.unsqueeze(1)
        dones         = dones.unsqueeze(1)       

        
        predicted_q = self.main_network(states) # forward pass through the main network to find the Q-values of the states
        predicted_q = predicted_q.gather(dim=1, index=actions) # selecting the Q-values of the actions that were actually taken

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():            
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0] # not argmax (cause we want the maxmimum q-value, not the action that maximize it)
            
        
        next_target_q_value[dones] = 0 
        y_js = rewards + (self.discount * next_target_q_value) # Compute the target Q-values
        loss = self.critertion(predicted_q, y_js) 
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss) 

            self.running_loss = 0
            self.learned_counts = 0
            
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        
        self.optimizer.step() # Update the parameters of the main network using the optimizer
 

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying 
        the parameters from the main network.
        """
        
        self.target_network.load_state_dict(self.main_network.state_dict())
   

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.
        
        """
        torch.save(self.main_network.state_dict(), path)
                  

class Model_TrainTest:
    def __init__(self, hyperparams):
        
        # Define RL Hyperparameters
        self.train_mode             = hyperparams["train_mode"]
        self.RL_load_path           = hyperparams["RL_load_path"]
        self.save_path              = hyperparams["save_path"]
        self.save_interval          = hyperparams["save_interval"]
        
        self.clip_grad_norm         = hyperparams["clip_grad_norm"]
        self.learning_rate          = hyperparams["learning_rate"]
        self.discount_factor        = hyperparams["discount_factor"]
        self.batch_size             = hyperparams["batch_size"]
        self.update_frequency       = hyperparams["update_frequency"]
        self.max_episodes           = hyperparams["max_episodes"]
        self.max_steps              = hyperparams["max_steps"]
        self.render                 = hyperparams["render"]
        
        self.memory_capacity        = hyperparams["memory_capacity"]
        
        self.num_states             = hyperparams["num_states"]
        self.render_fps             = hyperparams["render_fps"]
                        
        # Define Env
        self.env = gym.make('CartPole-v1', 
                            max_episode_steps=self.max_steps, 
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps # For max frame rate make it 0
        
        # Define the agent class
        self.agent = DQN_Agent(env                = self.env, 
                                clip_grad_norm    = self.clip_grad_norm,
                                learning_rate     = self.learning_rate,
                                discount          = self.discount_factor,
                                memory_capacity   = self.memory_capacity)
                
        
    def state_preprocess(self, state:int, num_states:int):
        
        normalized_state = torch.as_tensor(state, dtype=torch.float32, device=device)
    
        return normalized_state
    
    
    def train(self): 
        
        total_steps = 0
        self.reward_history = []
        
        # Training loop over episodes
        for episode in range(1, self.max_episodes+1):
            state, _ = self.env.reset(seed=seed)
            state = self.state_preprocess(state, num_states=self.num_states)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
                                                
            while not done and not truncation:
                action = self.agent.select_action(state, 0.07)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_state = self.state_preprocess(next_state, num_states=self.num_states)
                
                self.agent.replay_memory.store(state, action, next_state, reward, done) 
                
                if len(self.agent.replay_memory) > self.batch_size and sum(self.reward_history) > 0:
                    self.agent.learn(self.batch_size, (done or truncation))
                
                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()
                
                state = next_state
                episode_reward += reward
                step_size +=1
                            
            # Appends for tracking history
            self.reward_history.append(episode_reward) # episode reward                        
            total_steps += step_size
                                                                           
                            
            #-- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                # if episode != self.max_episodes:
                #     self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')
    
            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      )
            print(result)
        self.plot_training(episode)
                                                                    

    def test(self, max_episodes):  
        """                
        Reinforcement learning policy evaluation.
        """
           
        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()
        
        # Testing loop over episodes
        for episode in range(1, max_episodes+1):         
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
                                                           
            while not done and not truncation:
                state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state, 0.01)
                next_state, reward, done, truncation, _ = self.env.step(action)
                                
                state = next_state
                episode_reward += reward
                step_size += 1
                                                                                                                       
            # Print log            
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)
            
        pygame.quit() # close the rendering window
        
    
    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close() 
        
                
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()      
 
        

if __name__ == '__main__':
    # Parameters:
    train_mode = False
    render = not train_mode

    RL_hyperparams = {
        "train_mode"            : train_mode,
        "RL_load_path"          : f'./final_weights' + '_' + '3000' + '.pth',
        "save_path"             : f'./final_weights',
        "save_interval"         : 500,
        
        "clip_grad_norm"        : 3,
        "learning_rate"         : 12e-4,
        "discount_factor"       : 0.97,
        "batch_size"            : 32,
        "update_frequency"      : 10,
        "max_episodes"          : 3000           if train_mode else 5,
        "max_steps"             : 500,
        "render"                : render,
    
        
        "memory_capacity"       : 2_500        if train_mode else 0,
            
        "num_states"            : 4,
        "render_fps"            : 20,
        }
    
    
    # Run
    DRL = Model_TrainTest(RL_hyperparams) # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])
