# Based on https://arxiv.org/pdf/2406.08414
from dataclasses import dataclass
import linecache
import math
import traceback
from typing import *

from bs4 import BeautifulSoup
import torch
from torch import nn
import torch.nn.functional as F
import transformers
import numpy as np

from babylm.search_ideas.llms.anthropic import prompt_llm


PROMPT_PREFIX = """"
You're a machine learning engineer, mathematician, and evolutionary neurobiologist
who is researching linear-complexity causal self-attention algorithms for transformer models.

When you respond, output an XML document "<proposal>" where the
first section ("<thought>") corresponds to your thought process when
designing the next function. The other section ("<code>")
corresponds to the exact python code that you would like to try.
Here are some examples to get started:

<proposal name="mult_cumsum">
<thought>
A simple place to start out.
</thought>
<code>
def mult_cumsum_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = keys * values  # bind keys and values
    kvt = kv.cumsum(dim=2)  # cumsum over sequence is causal
    return kvt * queries  # retrieve queried values at each step
</code>
</proposal>

<proposal name="hrr">
<thought>
Try using holographic reduced representations for a key-value query.
</thought>
<code>
def hrr_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = torch.fft.rfft(keys) * torch.fft.rfft(values)  # bind keys and values
    kvt = kv.cumsum(dim=2)  # cumsum over sequence is causal
    return torch.fft.irfft(kvt * torch.fft.rfft(queries))  # retrieve queried values at each step
</code>
</proposal>

You must use the exact function interface used above. Feel free to
define extra hyperparameters within your function as constants.

Important Requirements:
 * Make sure your self-attention algorithm is both causal and sub-quadratic.
 * The input tensors all have shape (batch_size, num_heads, sequence_length, head_dims)
 * Keep track of the dimensions you are using to prevent shape errors.
 * You cannot define new model layers/parameters. The key, query, value vectors are already projections of the residual stream.
 * Avoid loops as much as possible. Vectorize your ops.
 * Never do a for loop over the sequence length.
 * Do not repeat experiments.

After a training run, the user will then return to you a fitness that corresponds to the
loss of the resulting model on the downstream task.
Your goal is to minimize loss.
Output valid, properly escaped XML only without additional commentary.
""".strip()


def make_history_prompt(history):
    sep = '=' * 10
    prompt = [sep]
    for proposal in history:
        prompt += [
            proposal.raw,
            sep
        ]
        if proposal.error is not None:
            prompt += [
                'Code not valid. Error:',
                proposal.error
            ]
        else:
            prompt.append('Loss:' + str(proposal.loss))
        prompt += [
            'Please generate the next one.',
            sep
        ]
    return '\n'.join(prompt)


def get_generated_function(str_func):
    before_exec = set(locals().keys())
    exec(str_func)
    after_exec = set(locals().keys())
    new_vars = after_exec - before_exec
    for name in new_vars:
        var = locals()[name]
        if callable(var):
            return var


def propose_code(history):
    num_errors_remaining = 3
    while num_errors_remaining:
        history_prompt = make_history_prompt(history)
        prompt = PROMPT_PREFIX + '\n' + history_prompt
        print('PROMPT', '=' * 30)
        response_prefix = '<proposal name="'
        stop_sequence = '</proposal>'
        raw_proposal = prompt_llm(
            prompt,
            response_prefix=response_prefix,
            stop_sequences=[stop_sequence]
        ).strip()
        if not raw_proposal.startswith(response_prefix):
            raw_proposal = response_prefix + raw_proposal
        if not raw_proposal.endswith(stop_sequence):
            raw_proposal += stop_sequence
        print('RAW PROP', '=' * 30)
        print(raw_proposal)
        result = Proposal.from_raw(raw_proposal)
        if result.error is not None:
            print('Error creating proposal:', result.error)
            history.append(result)
            num_errors_remaining -= 1
        else:
            break
    if num_errors_remaining <= 0:
        raise Exception('Too many errors!')
    return result


@dataclass
class Proposal:
    raw: str
    code: str = ''
    loss: float = np.inf
    error: str = None

    @property
    def func(self):
        return get_generated_function(self.code)

    @classmethod
    def from_raw(self, raw):
        try:
            doc = BeautifulSoup(raw, 'lxml')
            code = doc.find('code').get_text().strip()
        except Exception as e:
            error = str(e)
            proposal = Proposal(raw=raw, error=error)
        else:
            proposal = Proposal(raw=raw, code=code)
        # test parse func:
        if proposal.error is None:
            try:
                func = proposal.func
            except Exception as e:
                proposal.error = str(e)
            else:
                proposal.error = test_attention_func(func, code)
        return proposal


@torch.no_grad()
def test_attention_func(f_attn, raw_code):
    batch_size, seq_len, num_heads, model_dims = 2, 100, 8, 96
    k, v, q = torch.randn(batch_size, num_heads, seq_len, 3 * model_dims).split(model_dims, dim=-1)
    print(k.shape)
    try:
        result = f_attn(k, v, q)
    except Exception as e:
        # get the relevant code from the traceback
        tb = e.__traceback__
        tb_lines = []
        extracted_tb = traceback.extract_tb(tb)
        for frame in extracted_tb:
            if frame.filename == '<string>':
                code_line = raw_code.split('\n')[frame.lineno - 1].strip()
                tb_lines.append(f'In your code, line {frame.lineno}: {code_line}')
                break
            else:
                code_line = linecache.getline(frame.filename, frame.lineno).strip()
                tb_lines.append(f'In external file: {code_line}')

        error_message =  str(e) + '\n' + '\n'.join(tb_lines)
        return error_message
    return None


if __name__ == '__main__':
#     testcode = """
# def prune_func(model: torch.nn.Module):
#     for name, param in model.named_parameters():
#         importance = torch.abs(param.data * param.grad)
#         threshold = torch.quantile(importance, 0.2)
#         mask = (torch.abs(param) > threshold).float()
#         param.mul_(mask)
#     """
#     prune_func = get_generated_function(testcode)
#     print(prune_func)

#     doc = """
# <proposal name="abs_mag">
# <thought>
# This is a simple place to start: prune the smallest weights.
# </thought>
# <code>
# def mult_cumsum_attention(
#     keys: torch.Tensor,
#     values: torch.Tensor,
#     queries: torch.Tensor,
# ):
#     kv = keys * values  # bind keys and values
#     kvt = kv.cumsum(dim=1)  # cumsum over sequence is causal
#     return kvt * queries  # retrieve queried values at each step
# </code>
# </proposal>
#     """.strip()
#     xdoc = BeautifulSoup(doc, 'lxml')
#     print(xdoc.find('code').get_text())

    badstr = """
def bad_func(keys, values, queries):
    kv = keys * values.permute_dims(0, 2, 1, 3)  # bind keys and values
    kvt = kv.cumsum(dim=1)  # cumsum over sequence is causal
    return kvt * queries  # retrieve queried values at each step
    """.strip()
    badfunc = get_generated_function(badstr)
    msg = test_attention_func(badfunc, badstr)
    print(msg)

    badstr = """
def bad_func(keys, values, queries):
    x = torch.einsum('abc,def->q', keys, values)
    kvt = kv.cumsum(dim=1)  # cumsum over sequence is causal
    return kvt * queries  # retrieve queried values at each step
    """.strip()
    badfunc = get_generated_function(badstr)
    msg = test_attention_func(badfunc, badstr)
    print(msg)