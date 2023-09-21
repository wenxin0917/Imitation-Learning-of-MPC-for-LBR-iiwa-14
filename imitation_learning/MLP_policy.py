import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

class MLPPolicy(nn.Module):
    
    def __init__(self, input_dim, output_dim,n_layers,size,device,lr,
                 training=True,**kwaargs):
        # device CPU or GPU
        # training indicate whether the model is in trianing mode or not
        super().__init__()
        # init vars
        self.device = device
        self.training = training
        # network architecture
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, size))#first hidden layer
        self.mlp.append(nn.ReLU())

        for h in range(n_layers - 1): #additional hidden layers
            self.mlp.append(nn.Linear(size, size))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(size, output_dim)) #output layer
        self.mlp.append(nn.Tanh())
        output_scales = [320, 320, 176, 176, 110, 40, 40]
        self.output_scales = nn.Parameter(torch.tensor(output_scales, dtype=torch.float32))
        
        # loss and optimizer
        if self.training:
            self.loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.parameters(),lr)
            
        
        self.to(self.device) # send the model to the device
        
        
    def forward(self,obs):
        for layer in self.mlp:
            obs = layer(obs)
        obs = obs * self.output_scales
        return obs
    
    # save the model's learnable parameters to a file, include the name and tensor of the parameters
    def save(self,filepath):
        torch.save(self.state_dict(),filepath)
    
    def restore(self,filepath):
        self.load_state_dict(torch.load(filepath))
        
    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        return self(torch.Tensor(observation).to(self.device)).cpu().detach().numpy()
        # the observation tensor is pasesed through the model 'self' by calling 'self(tensor)',
        # which invokes the model's forward function, the result is a tensor representing the predicted action
        
    
    def update(self, observations, actions):
        assert self.training, 'Policy must be created with training = true in order to perform training updates...'

        # define network update
        self.optimizer.zero_grad()
        predicted_actions = self(torch.Tensor(observations).to(self.device))
        loss = self.loss_fn(predicted_actions, torch.Tensor(actions).to(self.device))
        loss.backward()
        self.optimizer.step()
        # print("loss_in_batch_size:", loss.item())
        return loss.item()  
    
    def compute_loss(self,observations,actions):
        predicted_actions = self(torch.Tensor(observations).to(self.device))
        loss = self.loss_fn(predicted_actions, torch.Tensor(actions).to(self.device))
        return loss.item()