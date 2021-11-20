from torch import nn

class Retrieve(nn.Module):
    def __init__(self):
        super(Retrieve, self).__init__()
    
    def forward(data):
        return (data+1)/2