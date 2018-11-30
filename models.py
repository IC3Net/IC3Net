import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(MLP, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(num_inputs, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, info={}):
        x = self.tanh(self.affine1(x))
        h = self.tanh(sum([self.affine2(x), x]))
        v = self.value_head(h)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return [F.log_softmax(head(h), dim=-1) for head in self.heads], v


class Random(nn.Module):
    def __init__(self, args, num_inputs):
        super(Random, self).__init__()
        self.naction_heads = args.naction_heads

        # Just so that pytorch is happy
        self.parameter = nn.Parameter(torch.randn(3))

    def forward(self, x, info={}):

        sizes = x.size()[:-1]

        v = Variable(torch.rand(sizes + (1,)), requires_grad=True)
        out = []

        for o in self.naction_heads:
            var = Variable(torch.randn(sizes + (o, )), requires_grad=True)
            out.append(F.log_softmax(var, dim=-1))

        return out, v


class RNN(MLP):
    def __init__(self, args, num_inputs):
        super(RNN, self).__init__(args, num_inputs)
        self.nagents = self.args.nagents
        self.hid_size = self.args.hid_size
        if self.args.rnn_type == 'LSTM':
            del self.affine2
            self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)

    def forward(self, x, info={}):
        x, prev_hid = x
        encoded_x = self.affine1(x)

        if self.args.rnn_type == 'LSTM':
            batch_size = encoded_x.size(0)
            encoded_x = encoded_x.view(batch_size * self.nagents, self.hid_size)
            output = self.lstm_unit(encoded_x, prev_hid)
            next_hid = output[0]
            cell_state = output[1]
            ret = (next_hid.clone(), cell_state.clone())
            next_hid = next_hid.view(batch_size, self.nagents, self.hid_size)
        else:
            next_hid = F.tanh(self.affine2(prev_hid) + encoded_x)
            ret = next_hid

        v = self.value_head(next_hid)
        if self.continuous:
            action_mean = self.action_mean(next_hid)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v, ret
        else:
            return [F.log_softmax(head(next_hid), dim=-1) for head in self.heads], v, ret

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

