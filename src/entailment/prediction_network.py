import torch.nn as nn


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        roberta_seq_len=256,
        roberta_hidden_state_len=768,
        tapas_seq_len=512,
        tapas_hidden_state_len=128,
    ):
        super(PredictionNetwork, self).__init__()
        output_dim = 3  # there are 3 different labels
        # Should be 262144 with default values
        input_dim = (
            roberta_seq_len * roberta_hidden_state_len
            + tapas_seq_len * tapas_hidden_state_len
        )

        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
