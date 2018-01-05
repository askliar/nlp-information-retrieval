# ------- Import -------
import torch as th
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad
from torch.autograd import Variable as V

import numpy as np

# ------- Setup -------
th.manual_seed(42)

batch_size = 4
max_seq_len = 5
feature_dim = 5
hidden_dim = 3

seq_lengths = [4, 3, 2, 1]

# model and data
lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
x = V(th.randn(batch_size, max_seq_len, feature_dim))

# "padding"
for i, idx in enumerate(seq_lengths):
    x[i, idx:, :] = 0

# ------- Using PackedSequence -------
x_packed = pack(x, seq_lengths, batch_first=True)
_, (last_states_packed, _) = lstm(x_packed)

print(last_states_packed.squeeze()[0, ...])  # This is a last state for first sequence in batch.
# This means it is the state after 4 LSTM steps, since this sequence was of length 4

# Variable containing:
#  0.1615
# -0.0783
# -0.3117
# [torch.FloatTensor of size 3]


# ------- Using a raw, padded Tensors -------
full_output, (last_states, _) = lstm(x)

print(last_states.squeeze()[
          0, ...])  # This time this is not what we want! This is after 5 LSTM steps (max sequence length)

# Variable containing:
#  0.0894
# -0.0987
# -0.3530
# [torch.FloatTensor of size 3]

print(full_output[0, seq_lengths[0] - 1, :])  # This is what we want. What we got above was full_output[0, -1, :].

# Variable containing:
#  0.1615
# -0.0783
# -0.3117
# [torch.FloatTensor of size 3]


# ------- Extract data using gather -------
seq_end_idx = V(th.LongTensor(seq_lengths) - 1, requires_grad=False)
seq_end_idx_ex = seq_end_idx.view(-1, 1, 1).expand(-1, 1, hidden_dim)

last_states_sliced = full_output.gather(1, seq_end_idx_ex)

assert np.allclose(last_states_sliced.data.squeeze().numpy(), last_states_packed.data.squeeze().numpy())

# ------- Extract data using advanced indexing -------

row_indices = th.arange(0, batch_size).long()
last_states_indexed = full_output[row_indices, seq_end_idx, :]

assert np.allclose(last_states_indexed.data.squeeze().numpy(), last_states_packed.data.squeeze().numpy())

