import torch
from torch import nn


def make_encoder(input_dim, enc_dec_dims):
    encoder_layers = []
    decoder_layers = []
    output_dim = input_dim
    enc_shape = enc_dec_dims[-1]
    for enc_dim in enc_dec_dims[:-1]:
        encoder_layers.extend([nn.Linear(input_dim, enc_dim), nn.SELU()])
        input_dim = enc_dim

    encoder_layers.append(nn.Linear(input_dim, enc_shape))

    enc_dec_dims = list(reversed(enc_dec_dims))
    for dec_dim in enc_dec_dims[1:]:
        decoder_layers.extend([nn.Linear(enc_shape, dec_dim), nn.SELU()])
        enc_shape = dec_dim

    decoder_layers.append(nn.Linear(enc_shape, output_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class FsrFgModel(nn.Module):
    def __init__(self, fg_input_dim, mfg_input_dim, num_input_dim, enc_dec_dims, output_dims,
                 num_tasks, dropout, method):
        super(FsrFgModel, self).__init__()

        self.method = method
        if self.method == 'FG':
            input_dim = fg_input_dim
        elif self.method == 'MFG':
            input_dim = mfg_input_dim
        elif self.method == 'FGR':
            input_dim = fg_input_dim + mfg_input_dim
        else:
            input_dim = fg_input_dim + mfg_input_dim
        if self.method != 'FGR_desc':
            fcn_input_dim = enc_dec_dims[-1]
        else:
            fcn_input_dim = num_input_dim + enc_dec_dims[-1]
        self.encoder, self.decoder = make_encoder(input_dim, enc_dec_dims)
        self.dropout = nn.Dropout(dropout)
        self.predict_out_dim = num_tasks
        self.batch_norm = nn.BatchNorm1d(fcn_input_dim)

        layers = []
        for output_dim in output_dims:
            layers.extend([nn.Linear(fcn_input_dim, output_dim), nn.SELU(), nn.BatchNorm1d(output_dim)])
            fcn_input_dim = output_dim

        layers.extend([self.dropout, nn.Linear(fcn_input_dim, num_tasks)])

        self.predictor = nn.Sequential(*layers)

    def forward(self, fg=None, mfg=None, num_features=None):

        if self.method == 'FG':
            z_d = self.encoder(fg)
        elif self.method == 'MFG':
            z_d = self.encoder(mfg)
        elif self.method == 'FGR':
            z_d = self.encoder(torch.cat([fg, mfg], dim=1))
        else:
            z_d = self.encoder(torch.cat([fg, mfg], dim=1))

        v_d_hat = self.decoder(z_d)

        if self.method == 'FGR_desc':
            z_d = torch.cat([z_d, num_features], dim=1)

        output = self.predictor(z_d)
        return output, v_d_hat
