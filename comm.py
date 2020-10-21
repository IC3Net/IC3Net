import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)

        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.hid_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.hid_size, 1)

        if self.args.transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_size, nhead=2)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # self.self_attn = nn.MultiheadAttention(self.hid_size, 2, dropout=0.1)
        # # Implementation of Feedforward model
        # self.linear1_att = nn.Linear(self.hid_size, 2048)
        # self.dropout_att = nn.Dropout(0.1)
        # self.linear2_att = nn.Linear(2048, self.hid_size)
        #
        # self.dropout1_att = nn.Dropout(0.1)
        # self.dropout2_att = nn.Dropout(0.1)
        # self.ReLU_att = nn.ReLU()


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask_1_dim = agent_mask
        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).clone().unsqueeze(-1)

        return num_agents_alive, agent_mask, agent_mask_1_dim

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask, agent_mask_1_dim = self.get_agent_mask(batch_size, info) #Return vehs in env

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state

            if self.args.transformer:
                if num_agents_alive == 0:
                    comm_sum = torch.zeros(comm.size())
                else:
                    # print(agent_mask_1_dim)
                    agent_mask = agent_mask_1_dim.unsqueeze(-1)
                    agent_mask = agent_mask.expand_as(comm)
                    src_mask = agent_mask_1_dim.view(batch_size, n).int()
                    src_mask = src_mask ^ torch.ones(src_mask.size()).int()
                    # print(src_mask.int())

                    comm = comm * agent_mask
                    # comm_sum = self.att_block(comm.transpose(0, 1))
                    # print(comm)
                    comm_sum = self.transformer_encoder(comm.transpose(0, 1), src_key_padding_mask=src_mask)
                    comm_sum = comm_sum.transpose(0, 1)

                    comm_sum = comm_sum * agent_mask
                # print(comm_sum)
            else:
                # Get the next communication vector based on next hidden state
                comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

                # Create mask for masking self communication
                mask = self.comm_mask.view(1, n, n)
                mask = mask.expand(comm.shape[0], n, n)
                mask = mask.unsqueeze(-1)

                mask = mask.expand_as(comm)
                comm = comm * mask

                if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                    and num_agents_alive > 1:
                    comm = comm / (num_agents_alive - 1)

                # Mask comm_in
                # Mask communcation from dead agents
                comm = comm * agent_mask
                # Mask communication to dead agents
                comm = comm * agent_mask_transpose

                # Combine all of C_j for an ith agent which essentially are h_j
                comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    # def att_block(self, src):
    #     src2 = self.self_attn(src, src, src)[0]
    #     src = src + self.dropout1_att(src2)
    #     # src = self.norm1(src)
    #     src2 = self.linear2_att(self.dropout_att(self.ReLU_att(self.linear1_att(src))))
    #     src = src + self.dropout2_att(src2)
    #     # src = self.norm2(src)
    #     return src
