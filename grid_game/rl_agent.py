import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os


class Network(nn.Module):                                    #defines the model 
    def __init__(self,input_size,output_size):
        super(Network,self).__init__()                       #initialises the base class 
        self.fc1=nn.Linear(input_size,64)                    #state input to 64 neurons
        self.fc2=nn.Linear(64,64)                            #another 64 neurons
        self.fc3=nn.Linear(64,output_size)                   #output layer=no of actions

    def forward(self,x):                                     #defines now data moves through the network
        x=F.relu(self.fc1(x))                                #applies relu activation to introduce non-linearity
        x=F.relu(self.fc2(x))
        return self.fc3(x)                                   #output Q-values for each action
    
class ReplayMemory:                                          #stores past experiences
    def __init__(self,capacity):
        self.capacity=capacity                               #max size of memory
        self.memory=[]                                       #list storing transitions

    def push(self,transition):
        self.memory.append(transition)                       #adds a nwe experience
        if len(self.memory)>self.capacity:              
            del self.memory[0]                               #removes oldest one

    def sample(self,batch_size):                             #randomly select a batch of experiences 
        batch=random.sample(self.memory,batch_size)
        return zip(*batch)                                   #unzips into tuples...
    
    def __len__(self):                                       #how many experiences are stored..
        return len(self.memory)
        
class DQNAgent:
    def __init__(self,input_size,output_size,gamma=0.9,lr=1e-3,epsilon=1.0,epsilon_min=0.05,epsilon_decay=0.995):
        self.gamma=gamma                                     #discount factor for future rewards..
        self.epsilon=epsilon                                 #starting probability of random action
        self.epsilon_min=epsilon_min                         #min allowed epsilon
        self.epsilon_decay=epsilon_decay                     #how fast epsilon decreases
        self.input_size=input_size
        self.output_size=output_size
        self.model=Network(input_size,output_size)           #stores the parameters as class variables
        self.memory=ReplayMemory(100000)                     #stores upto 100000 past experiences
        self.optimizer=optim.Adam(self.model.parameters(),lr=lr)  #use adam optimizer to train the model
        self.criterion=nn.MSELoss()                          #use mean squared error between predicted Q-values and target Q-value

    def select_action(self,state):
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0)    #converts state to a pytorch tensor and adds batch dimension
        if random.random()<self.epsilon:                              #with probability epsilon choose a random action
            return random.randint(0,self.output_size-1)
        with torch.no_grad():                                         #otherwise select action with highest Q-value
            q_values=self.model(state)
            return q_values.argmax().item()
        
    def remember(self,state,action,reward,next_state,done):           #converts each element to pytorch tensors and stores them in memory
        self.memory.push((torch.tensor(state, dtype=torch.float32),
                          torch.tensor([action], dtype=torch.int64),
                          torch.tensor([reward], dtype=torch.float32),
                          torch.tensor(next_state, dtype=torch.float32),
                          torch.tensor([done], dtype=torch.bool)))
    
    def replay(self,batch_size):                                    #learn from past experiences
        if len(self.memory)<batch_size:
            return 
        states,actions,rewards,next_states,dones=self.memory.sample(batch_size)    #converts the sampled experience batch into proper tensors using torch.stack 
        states=torch.stack(states)
        actions=torch.stack(actions)
        rewards=torch.stack(rewards)
        next_states=torch.stack(next_states)
        dones=torch.stack(dones)

        q_values=self.model(states).gather(1,actions)    #predicts Q-values for all actions and picks the Q values for the actions taken
        with torch.no_grad():                            #predicts max Q-values for next state
            max_next_q_values=self.model(next_states).max(1)[0].unsqueeze(1)
            target_q_values=rewards+self.gamma*max_next_q_values*(~dones)

        loss=self.criterion(q_values,target_q_values)    #measures how close the predicted Q-values are to the target Q-values
        self.optimizer.zero_grad()                       #standard pytorch training loop
        loss.backward() 
        self.optimizer.step()

        if self.epsilon>self.epsilon_min:                #gradually reduces epsilon
            self.epsilon*=self.epsilon_decay


    def save(self,filename='dqn_model.pth'):             #save the model parameters 
        torch.save(self.model.state_dict(),filename)
        
    def load(self,filename='dqn_model.pth'):             #load model if file exists
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            print("model loaded")
        else:
            print("no model file found")
