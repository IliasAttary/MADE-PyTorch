import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    # W = (out_features, in_features)
    # b = (out_features,)

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # register_buffer so that mask is saved in state_dict but not trained
        self.register_buffer('mask', torch.ones(out_features, in_features)) 

    def set_mask(self, mask):
        mask = torch.as_tensor(mask, device=self.mask.device, dtype=self.mask.dtype)
        if mask.shape != self.mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match layer weight shape {self.mask.shape}.")
        self.mask.copy_(mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_sizes = hidden_dims

        # Build layers
        layers=[]

        layers.append(MaskedLinear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims)-1):
            layers.append(MaskedLinear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        # no activation on output layer
        layers.append(MaskedLinear(hidden_dims[-1], output_dim))

        self.layers=nn.ModuleList(layers)

        # Build masks
        self._init_masks()

        

    def _init_masks(self):
        D = self.input_size
        masked_layers = [layer for layer in self.layers if isinstance(layer, MaskedLinear)]
        device = masked_layers[0].weight.device

        # Assign a degree to each neuron
        m_in = torch.arange(1, D + 1, device=device)

        degrees = []
        prev = m_in

        for h in self.hidden_sizes:
            low = int(prev.min().item())
            m_h = torch.randint(low, D, (h,), device=device)  # [low, D-1]
            degrees.append(m_h)
            prev = m_h

        m_out = m_in

        # Create masks based on degrees
        degrees = [m_in] + degrees + [m_out]
        for l in range(len(masked_layers) - 1):
            m_prev = degrees[l]
            m_next = degrees[l + 1]
            mask = (m_next[:, None] >= m_prev[None, :]).float()  # (out, in)
            masked_layers[l].set_mask(mask)

        m_prev = degrees[-2]
        m_next = degrees[-1]
        mask = (m_next[:, None] > m_prev[None, :]).float()
        masked_layers[-1].set_mask(mask)
        
        

