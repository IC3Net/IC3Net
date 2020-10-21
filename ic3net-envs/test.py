import torch
import torch.nn as nn

class Communacation(nn.Module):
    def __init__(self, input_size, hid_size):
        super(Communacation, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(input_size, dim_feedforward = 128, nhead=2, dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, x, src_mask):
        x = self.encoder(x, src_key_padding_mask=src_mask)
        return x



if __name__ == "__main__":
    torch.manual_seed(0)
    com = Communacation(8, 128)
    x = torch.ones(1, 10, 8)
    x[:, [5, 6], :] = torch.zeros(8)
    mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]])
    x = x * mask.view(1, 10, 1)
    y = com(x.transpose(0, 1), mask ^ torch.ones(mask.size()).int()).transpose(0, 1)
    print(y)
