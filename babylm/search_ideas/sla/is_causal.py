import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from babylm.search_ideas.llms.openai import prompt_llm


PROMPT = """
You're a machine learning researcher exploring causal self-attention algorithms.
Given a function, evaluate whether it leaks information to future tokens thereby
violating the causal constraint.

```
%s
```

Respond as follows: If there is no violation output "CAUSAL.", else output "NOT CAUSAL.",
the specific code which causes the problem, and a one sentence explanation. No more and no less.
""".strip()


def check_is_causal(f_attn):
    is_causal, accuracy, loss = train_test_model(
        f_attn, batch_size=65, seq_len=64,
        num_tokens=100, model_dims=256, num_heads=4,
        max_iters=100, lr=0.001
    )
    return is_causal


def check_is_causal_llm(code):
    prompt = PROMPT % code
    response = prompt_llm(prompt).strip()
    if response.lower().startswith('causal'):
        return True, response
    else:
        return False, response


class TestModel(torch.nn.Module):
    def __init__(self, f_attn, num_tokens, model_dims, num_heads=4):
        super().__init__()
        self.f_attn = f_attn
        self.embedding = nn.Embedding(num_tokens, model_dims)
        self.output = nn.Linear(model_dims, num_tokens)
        self.model_dims = model_dims
        self.num_heads = num_heads
        self.head_dims = model_dims // num_heads
        self.qkv = nn.Linear(self.model_dims, 3 * self.model_dims, bias=False)

    def forward(self, x):
        z = self.embedding(x)
        batch_size, seq_len = z.shape[:2]
        q, k, v = self.qkv(z).split(self.model_dims, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dims)
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dims)
        k = k.permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dims)
        v = v.permute(0, 2, 1, 3)
        values_hat = self.f_attn(k, v, q)
        values_hat = values_hat.permute(0, 2, 1, 3).contiguous()
        values_hat = values_hat.view(batch_size, seq_len, self.model_dims)
        return self.output(values_hat)


def train_test_model(
    f_attn, batch_size=64, seq_len=64, num_tokens=100, model_dims=256, num_heads=4,
    max_iters=100, lr=0.01, clip_grad_norm=1.,
):
    random_accuracy = 1. / num_tokens
    model = TestModel(f_attn, num_tokens, model_dims, num_heads=num_heads)
    if torch.cuda.is_available():
        model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_causal = True
    for i in range(max_iters):
        optimizer.zero_grad()
        batch = torch.randint(0, num_tokens, (batch_size, seq_len + 1))
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            labels.contiguous().view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        accuracy = (logits.argmax(-1) == labels).sum() / np.prod(labels.shape)
        # Is accuracy too good at predicting the future random values?
        if accuracy > random_accuracy * 2:
            is_causal = False
            break
    del optimizer
    del model
    return is_causal, accuracy, loss


if __name__ == '__main__':
    def causal_attention(
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ):
        kv = keys * values  # bind keys and values
        kvt = kv.cumsum(dim=2)
        return kvt * queries  # retrieve queried values at each step

    def not_causal_attention(
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ):
        kv = keys * values  # bind keys and values
        kvt = kv.sum(dim=2, keepdim=True)
        return kvt * queries  # retrieve queried values at each step

    def adaptive_sparse_fourier_attention_v6(
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ):
        batch_size, num_heads, seq_len, head_dims = keys.shape
        device = keys.device

        def multi_head_sparse_attention(q, k, v):
            sparse_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
            sparse_mask[:, ::4] = True  # Attend to every 4th position
            sparse_mask.fill_diagonal_(True)  # Always attend to self

            scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(head_dims)
            scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        def fourier_attention(q, k, v):
            q_fft = torch.fft.rfft(q, dim=2)
            k_fft = torch.fft.rfft(k, dim=2)
            v_fft = torch.fft.rfft(v, dim=2)

            qk_fft = q_fft * k_fft.conj()
            attn_fft = qk_fft / (torch.abs(qk_fft) + 1e-8)
            output_fft = attn_fft * v_fft

            return torch.fft.irfft(output_fft, n=seq_len, dim=2)

        # Compute multi-head sparse attention
        sparse_output = multi_head_sparse_attention(queries, keys, values)

        # Compute Fourier attention
        fourier_output = fourier_attention(queries, keys, values)

        # Gating mechanism
        gate = torch.sigmoid(torch.einsum('bhid,bhid->bhi', queries, keys).unsqueeze(-1))

        # Adaptive combination of sparse and Fourier outputs
        combined_output = gate * sparse_output + (1 - gate) * fourier_output

        # Attention pooling for adaptive weights
        attn_pool = torch.einsum('bhid,bhid->bhi', combined_output, queries)
        adaptive_weights = torch.softmax(attn_pool, dim=1).unsqueeze(-1)

        # Apply adaptive weights
        output = torch.sum(combined_output * adaptive_weights, dim=1, keepdim=True)

        # Residual connection and layer normalization
        output = output + values
        output = F.layer_norm(output, (head_dims,))

        # Gradient clipping for stability
        output = torch.clamp(output, min=-10, max=10)

        return output

    print('training on bad generated function')
    is_causal, accuracy, loss = train_test_model(
        adaptive_sparse_fourier_attention_v6, batch_size=64, seq_len=64,
        num_tokens=100, model_dims=256, num_heads=4,
        max_iters=1000, lr=0.001
    )
    assert not is_causal

    print('training on good function')
    is_causal, accuracy, loss = train_test_model(
        causal_attention, batch_size=64, seq_len=64,
        num_tokens=100, model_dims=256, num_heads=4,
        max_iters=100, lr=0.001
    )
    assert is_causal

    print('training on bad function')
    is_causal, accuracy, loss = train_test_model(
        not_causal_attention, batch_size=64, seq_len=64,
        num_tokens=100, model_dims=256, num_heads=4,
        max_iters=100, lr=0.001
    )
    assert not is_causal
