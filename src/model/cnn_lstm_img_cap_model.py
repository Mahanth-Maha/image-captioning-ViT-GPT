import torch.nn as nn
from encoder.cnn_encoder import CNNEncoder
from decoder.lstm_decoder import LSTMDecoder

class CNNLSTMCaptioner(nn.Module):
    def __init__(self, vocab_size, cfg, pad_idx=None):
        super().__init__()
        self.encoder = CNNEncoder(
            cfg.cnn_name, 
            cfg.proj_dim, 
            cfg.cnn_train_backbone
        )
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            proj_dim=cfg.proj_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_idx=pad_idx
        )

    def forward(self, images, inp_ids):
        ctx = self.encoder(images)
        logits = self.decoder(ctx, inp_ids)
        return logits

    def generate(self, images, sos_id, eos_id, max_len):
        ctx = self.encoder(images)
        return self.decoder.greedy_decode(ctx, sos_id, eos_id, max_len)
