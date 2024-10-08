# Based on https://arxiv.org/pdf/2406.08414
from dataclasses import dataclass
import linecache
import math
import re
import traceback
from typing import *

from bs4 import BeautifulSoup
import torch
from torch import nn
import torch.nn.functional as F
import transformers
import numpy as np

from babylm.search_ideas.llms.anthropic import prompt_llm
from babylm.search_ideas.proposer import Proposal
from babylm.search_ideas.sla.is_causal import check_is_causal


PROMPT_PREFIX = """"
You're a machine learning engineer, mathematician, and evolutionary neurobiologist
who is researching causal self-attention algorithms for transformer models.

When you respond, output an XML document "<proposal>" where the
first section ("<thought>") corresponds to your thought process when
designing the next function. The other section ("<code>")
corresponds to the exact python code that you would like to try.
Here are some examples to get started:

Good example:
<proposal name="mult_cumsum">
<thought>
Start simple with a multiplicative key/value store and query.
</thought>
<code>
def mult_cumsum_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = keys * values  # bind keys and values
    kvt = kv.cumsum(dim=2)
    return kvt * queries  # retrieve queried values at each step
</code>
</proposal>

You must use the exact function interface used above. Feel free to
define extra hyperparameters within your function as constants.
The input tensors keys, values, queries each have shape (batch_size, num_heads, sequence_length, head_dims)

Important Requirements:
 * No forward information leakage i.e. step T may only attend to steps <= T
 * Keep track of the dimensions you are using to prevent shape errors.
 * Ensure any tensors you create are assigned to the same device as `keys.device`
 * You cannot define new nn.Modules/nn.Parameters. The key, query, value vectors are already projections of the residual stream.
 * Avoid loops as much as possible. Vectorize your ops.
 * Never use a for loop over the sequence length.
 * Leverage knowledge from previous experiments.
 * Experiment with novel techniques, possibly inspired from other fields of study like physics.
 * Do not repeat experiments.

After a training run, the user will then return to you a fitness that corresponds to the
loss of the resulting model on the downstream task.
Your goal is to minimize loss.
Output valid, properly escaped XML only without additional commentary.
""".strip()


@dataclass
class SLAProposal(Proposal):
    loss: float = np.inf

    @property
    def fitness_string(self):
        return f'Loss: {self.loss}'

    def test(self):
        error = test_attention_func(self.func, self.code)
        if error:
            self.error = error
        if not self.error:
            print('Check is_causal..')
            try:
                is_causal = check_is_causal(self.func)
            except Exception as e:
                # get the relevant code from the traceback
                tb = e.__traceback__
                tb_lines = []
                extracted_tb = traceback.extract_tb(tb)
                for frame in extracted_tb:
                    if frame.filename == '<string>':
                        code_line = self.code.split('\n')[frame.lineno - 1].strip()
                        tb_lines.append(f'In your code, line {frame.lineno}: {code_line}')
                        break
                    else:
                        code_line = linecache.getline(frame.filename, frame.lineno).strip()
                        tb_lines.append(f'In external file: {code_line}')

                error_message =  'Precheck failed' + str(e) + '\n' + '\n'.join(tb_lines)
                self.error = error_message
            else:
                if not is_causal:
                    self.error = "This self-attention function is not causal."

    @classmethod
    def prompt_prefix(cls):
        return PROMPT_PREFIX


def propose_code(history):
    proposal = SLAProposal.propose_code(history)
    return proposal


@torch.no_grad()
def test_attention_func(f_attn, raw_code):
    batch_size, seq_len, num_heads, model_dims = 2, 100, 8, 96
    k, v, q = torch.randn(batch_size, num_heads, seq_len, 3 * model_dims).split(model_dims, dim=-1)

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

    if has_seq_loop(raw_code):
        return 'Explicit loop over the sequence is not allowed. Vectorize the operations.'

    return None


re_seqloop = re.compile(r'for .+ in range\(.*(seq|l|t).*\)')
def has_seq_loop(code):
    result = re_seqloop.search(code)
    return bool(result)


if __name__ == '__main__':
    bad_code = """
def causal_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = keys * values  # bind keys and values
    kvt = kv.cumsum(dim=2)
    for i in range(seq_len):
        something_dumb()
    return kvt * queries  # retrieve queried values at each step
    """

    good_code = """
def causal_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = keys * values  # bind keys and values
    kvt = kv.cumsum(dim=2)
    return kvt * queries  # retrieve queried values at each step
    """
    code_has_seq_loop = has_seq_loop(bad_code)
    assert code_has_seq_loop

    code_has_seq_loop = has_seq_loop(good_code)
    assert not code_has_seq_loop

