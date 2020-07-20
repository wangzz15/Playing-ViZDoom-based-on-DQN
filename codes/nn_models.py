import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    """
    Description
    --------------
    Compute the output dimension when applying a convolutional layer.

    Parameters
    --------------
    size        : Int, width or height of the input.
    kernel_size : Int, the kernel size of the conv layer (default=5)
    stride      : Int, the stride used in the conv layer (default=2)
    """

    return (size - (kernel_size - 1) - 1) // stride + 1


class DQNetwork(nn.Module):

    def __init__(self, w=40, h=30, c=3, out=3, init_zeros=False, dueling=False):
        """
        Description
        ---------------
        Constructor of Deep Q-network class.

        Parameters
        ---------------
        w          : Int, input width
        h          : Int, input height
        c          : Int, input channels
        init_zeros : Boolean, whether to initialize the weights to zero or not.
        out        : Int, the number of output units, it corresponds to the number of possible actions (default=3).
                     Be careful, it must be changed when considering a different number of possible actions.
        """

        super(DQNetwork, self).__init__()
        self.dueling = dueling
        # Conv Module
        self.conv_1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        if init_zeros:
            nn.init.constant_(self.conv_1.weight, 0.0)
            nn.init.constant_(self.conv_2.weight, 0.0)

        # width of last conv output
        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
        # height of last conv output
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
        linear_input_size = convw * convh * 64
        if not self.dueling:
            self.fc = nn.Linear(linear_input_size, 128)
            self.output = nn.Linear(128, out)
        else:
            self.fc_value_1 = nn.Linear(linear_input_size, 128)
            self.fc_value_2 = nn.Linear(128, 1)
            self.fc_advantage_1 = nn.Linear(linear_input_size, 128)
            self.fc_advantage_2 = nn.Linear(128, out)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        if not self.dueling:            
            x = F.relu(self.fc(x.view(x.size(0), -1)))
            return self.output(x)
        else:
            x1 = F.relu(self.fc_value_1(x.view(x.size(0), -1)))
            x1 = self.fc_value_2(x1)
            x2 = F.relu(self.fc_advantage_1(x.view(x.size(0), -1)))
            x2 = self.fc_advantage_2(x2)
            return x1 + x2 - torch.mean(x2)


def test():
    a = torch.zeros((1, 3, 30, 40))
    Q = DQNetwork(c=3, dueling=True)
    print(Q(a))

if __name__ == "__main__":
    test()


