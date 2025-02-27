
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number

class AttnMeanPoolMIL(nn.Module):

    def __init__(self,
                 gated=True,
                 encoder_dim=256,
                 attn_latent_dim=128,
                 dropout=0.5,
                 out_dim=1,
                 activation='softmax',
                 encoder=None,
                 warm_start=False):
        super().__init__()


        if gated:
            attention = GatedAttn(n_in=encoder_dim,
                                  n_latent=attn_latent_dim,
                                  dropout=dropout)

        else:
            attention = Attn(n_in=encoder_dim,
                             n_latent=attn_latent_dim,
                             dropout=dropout)

        self.warm_start = warm_start
        self.attention = attention

        self.activation = activation

        if self.warm_start:
            for param in self.attention.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(nn.Linear(encoder_dim, out_dim))

        if encoder is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = encoder

    def start_attention(self, freeze_encoder=True, **kwargs):
        """
        Turn on attention & freeze encoder if necessary
        """
        for param in self.attention.parameters():
            param.requires_grad = True

        self.warm_start = False

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        return True

    def attend(self, x, coords=None):
        """
        Given input, compute attention scores

        Returns:
        - attn_scores: tensor (n_batches, n_instances, n_feats) Unnormalized attention scores
        """

        n_batches, n_instances, _ = x.shape
        if self.warm_start:
            attn_scores = torch.ones((n_batches, n_instances, 1), device=x.device)
        else:
            attn_scores = self.attention(x) # (n_batches * n_instances, 1)

        attn_scores = attn_scores.view(n_batches, n_instances, 1) # unflatten

        if self.activation == 'softmax':
            attn = F.softmax(attn_scores, dim=1) #(n_batches, n_instances, 1)
        else:
            attn = F.relu(attn_scores)

        bag_feats = x.view(n_batches, n_instances, -1)  # unflatten

        out = (bag_feats * attn).sum(1)

        return attn_scores, out

    def encode(self, x):
        x_enc = self.encoder(x)
        return x_enc

    def forward(self, x, coords):
        """
        Args:
        - x (n_batches x n_instances x feature_dim): input instance features
        - coords (list of tuples): coordinates. Not really used for this vanilla version

        Returns:
        - out (n_batches x encode_dim): Aggregated features
        - attn_dict (dict): Dictionary of attention scores
        """
        n_batches, n_instances, _ = x.shape
        attn_dict = {'inter': [], 'intra': []}

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc)  # (n_batches * n_instances, 1), (n_batches, encode_dim)

        out = self.head(out)

        attn_dict['intra'] = attn.detach()
        levels = torch.unique(coords[:, 0])
        attn_dict['inter'] = torch.ones(n_batches, len(levels), 1)    # All slices receive same attn

        return out, attn_dict

    def captum(self, x):
        """
        For computing IG scores with captum. Very similar to forward function
        """
        n_batches, n_instances, _ = x.shape

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc)  # (n_batches * n_instances, 1), (n_batches, encode_dim)

        out = self.head(out)

        return out


class Attn(nn.Module):
    """
    The attention mechanism from Equation (8) of (Ilse et al, 2008).

    Args:
    - n_in (int): Number of input dimensions.
    - n_latent (int or None): Number of latent dimensions. If None, will default to (n_in + 1) // 2.
    - dropout: (bool, float): Whether or not to use dropout. If True, will default to p=0.25

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        # basic attention scoring module
        self.score = [nn.Linear(n_in, n_latent),
                      nn.Tanh(),
                      nn.Linear(n_latent, 1)]

        # maybe add dropout
        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25

            self.score.append(nn.Dropout(p))

        self.score = nn.Sequential(*self.score)

    def forward(self, x):
        """
        Outputs normalized attention.

        Args:
        - x (n_batches, n_instances, n_in) or (n_instances, n_in): The bag features.

        Returns:
        - attn_scores (n_batches, n_instances, 1) or (n_insatnces, 1):
            The unnormalized attention scores.

        """
        attn_scores = self.score(x)

        return attn_scores


class GatedAttn(nn.Module):
    """
    The gated attention mechanism from Equation (9) of (Ilse et al, 2008).
    Parameters
    ----------
    n_in: int
        Number of input dimensions.
    n_latent: int, None
        Number of latent dimensions. If None, will default to (n_in + 1) // 2.
    dropout: bool, float
        Whether or not to use dropout. If True, will default to p=0.25
    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        self.tanh_layer = [nn.Linear(n_in, n_latent),
                           nn.Tanh()]

        self.sigmoid_layer = [nn.Linear(n_in, n_latent),
                              nn.Sigmoid()]

        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25

            self.tanh_layer.append(nn.Dropout(p))
            self.sigmoid_layer.append(nn.Dropout(p))

        self.tanh_layer = nn.Sequential(*self.tanh_layer)
        self.sigmoid_layer = nn.Sequential(*self.sigmoid_layer)

        self.w = nn.Linear(n_latent, 1)

    def forward(self, x):
        """
        Outputs normalized attention.

        Args:
        - x (n_batches, n_instances, n_in) or (n_instances, n_in): The bag features.

        Returns:
        - attn_scores (n_batches, n_instances, 1) or (n_insatnces, 1):
            The unnormalized attention scores.
        """

        attn_scores = self.w(self.tanh_layer(x) * self.sigmoid_layer(x))

        return attn_scores