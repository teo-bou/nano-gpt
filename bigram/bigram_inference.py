import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

device = "mps"


class Encoder_decoder:
    def __init__(self, chars):

        self.string_to_integer = {ch: i for i, ch in enumerate(chars)}
        self.integer_to_string = {i: ch for i, ch in enumerate(chars)}

    def encode(self, string):
        return [self.string_to_integer[c] for c in string]

    def decode(self, liste):
        return "".join(self.integer_to_string[i] for i in liste)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # (Batch, Block_size) for the idx, as well of target (B,T)
        logits = self.token_embedding_table(
            idx
        )  # (B,T,C) (Batch, Block_size, Vocab_size) in output - each token get a list of proba for the next token (a row of the embedding table)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C) with proba
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # Sample the next token (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # Append next, to do it again (B, T+1)
        return idx


with open("bigrame_encoder.model", "rb") as f:
    encoder_decoder = pickle.load(f)

m = torch.load("bigram.model", weights_only=False)


def generate(model, n, string=None):
    if string is None:
        return encoder_decoder.decode(
            m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), n)[0].tolist()
        )
    else:
        return encoder_decoder.decode(
            m.generate(
                torch.tensor(encoder_decoder.encode(string), dtype=torch.long)
                .view(1, -1)
                .to(device),
                n,
            )[0].tolist()
        )


print(generate(m, 1000, input()))
