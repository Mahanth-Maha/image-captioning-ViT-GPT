import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, proj_dim=512, num_layers=1, dropout=0.1, pad_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim + proj_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, img_ctx, inp_ids):
        B, T = inp_ids.shape
        emb = self.drop(self.embed(inp_ids))
        ctx = img_ctx.unsqueeze(1).expand(B, T, img_ctx.size(-1))
        x = torch.cat([emb, ctx], dim=-1)
        h, _ = self.lstm(x)
        h = self.drop(h)
        logits = self.fc(h)
        return logits

    @torch.no_grad()
    def greedy_decode(self, img_ctx, sos_id, eos_id, max_len=32):
        B = img_ctx.size(0)
        inputs = torch.full((B, 1), sos_id, dtype=torch.long, device=img_ctx.device)
        outputs = [inputs]
        state = None
        for _ in range(max_len-1):
            emb = self.embed(inputs)
            x = torch.cat([emb, img_ctx.unsqueeze(1)], dim=-1)
            h, state = self.lstm(x, state)
            logits = self.fc(h)
            nxt = logits.argmax(-1)
            outputs.append(nxt)
            inputs = nxt
        out = torch.cat(outputs, dim=1)

        if eos_id is not None:
            for b in range(B):
                eos_pos = (out[b] == eos_id).nonzero(as_tuple=False)
                if len(eos_pos) > 0:
                    out[b, eos_pos[0].item()+1:] = eos_id
        return out
