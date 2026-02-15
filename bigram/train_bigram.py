import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# ---
torch.manual_seed(42)
batch_size = 32
block_size = 8
epochs = 100000
cut = 1500000
device = "mps"
# ---

with open("dataset/text_dataset.txt") as dataset:
    text = dataset.read()
text = text[:cut]
chars = sorted(set(text))
vocab_size = len(chars)

string_to_integer = {ch: i for i, ch in enumerate(chars)}
integer_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(string):
    return [string_to_integer[c] for c in string]


def decode(liste):
    return "".join(integer_to_string[i] for i in liste)


data = torch.tensor(encode(text), dtype=torch.long)
data = data.to(device)
n = int(0.9 * len(data))
train_data = data[:n]  # TODO Improve because the data is heterogenous
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


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


m = BigramLanguageModel(vocab_size)
m = m.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for steps in tqdm(range(epochs)):
    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 1000 == 0:
        print(loss.item())

torch.save(m, "bigram.model")
print(
    decode(
        m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), 500)[0].tolist()
    )
)
