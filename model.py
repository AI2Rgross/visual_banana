import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=34, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
 #      States have shape: (1, 3,84, 84)

        self.cv1 = nn.Conv2d(in_channels=3, out_channels=12,kernel_size=3,stride=2, padding=0)
        self.RL1 = nn.ReLU(True)        
     
        self.cv2 = nn.Conv2d(in_channels=12,out_channels=6,kernel_size=2,stride=2, padding=0)
        self.RL2 = nn.ReLU(True)     
 
        self.fc1 = nn.Linear(6*20*20, action_size)        
   #     self.fc2 = nn.Linear(fc2_units, action_size)
 

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.cv1(state)
        #print("test",x.shape)
        
        x = F.relu(x)
        x = self.cv2(x)
        #print("test",x.shape)

        x = F.relu(x)
        #print("test",x.shape)
        x = x.view(x.size(0), -1)       
        x = self.fc1(x)
 
 
        return x

