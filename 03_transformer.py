import sys
sys.path.append('..')
import torch
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from minbpe import BasicTokenizer
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F



tokenizer = BasicTokenizer()
tokenizer.load(model_file="output/tokenizer/my_tokenizer.model")


def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)


torch.manual_seed(3647)

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = get_vocab_size(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedFoward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_tokens: Tensor of token indices of shape (batch_size, sequence_length)
            targets: Optional tensor of target token indices of same shape as input_tokens

        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, sequence_length, vocab_size)
            and loss is optional cross-entropy loss if targets are provided
        """

        B, T = input_tokens.shape

        # input_tokens and targets are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(input_tokens)  # (B,T,C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
                Generate new tokens given a context.

                Args:>ns: Starting token indices of shape (batch_size, sequence_length)
                        max_new_tokens: Number of new tokens to generate

                Returns:
                        Tensor of token indices of shape (batch_size, sequence_length + max_new_tokens)
                """

        # input_tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop input_tokens to the last block_size tokens
            cropped_input = input_tokens[:, -block_size:]
            # get the predictions
            logits, _ = self(cropped_input)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)  # (B, T+1)
        return input_tokens

# Implementation

# Parameters and Dummy INput

model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


batch_size = 1
seq_length = 6
x = torch.randint(0, vocab_size, (batch_size, seq_length))
x = x.to(device)

logits, loss = model(x)
print(logits.shape, loss)

# Display the model summary

def print_model_structure(model: torch.nn.Module, indent: str = '') -> None:
    """
    Custom function to print model structure in a hierarchical format
    """
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{indent}├─ {name}: {child.__class__.__name__} ({params:,} parameters)")
        print_model_structure(child, indent + '│  ')


print_model_structure(model)




def get_model_stats(model: torch.nn.Module) -> pd.DataFrame:
    """
    Create a DataFrame with detailed layer statistics
    """
    stats = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            stats.append({
                'Layer Name': name,
                'Type': module.__class__.__name__,
                'Parameters': params,
                'Trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
    return pd.DataFrame(stats)


stats_df = get_model_stats(model)
print(stats_df)
