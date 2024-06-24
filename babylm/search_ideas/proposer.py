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


@dataclass
class Proposal:
    raw: str
    code: str = ''
    error: str = None

    def test(self):
        pass

    @property
    def fitness_string(self):
        raise NotImplementedError(f'Define `fitness_string` on {self.__class__.__name__}')

    @property
    def func(self):
        return get_generated_function(self.code)

    @classmethod
    def from_raw(cls, raw):
        # parse raw proposal
        try:
            doc = BeautifulSoup(raw, 'lxml')
            code = doc.find('code').get_text().strip()
        except Exception as e:
            # parsing error happened
            error = str(e)
            proposal = cls(raw=raw, error=error)
        else:
            proposal = cls(raw=raw, code=code)
        # test that a function can be extracted:
        if proposal.error is None:
            try:
                func = proposal.func
            except Exception as e:
                proposal.error = str(e)
        return proposal

    @classmethod
    def prompt_prefix(cls):
        raise NotImplementedError(f'Define a prompt_prefix for {cls.__name__}')

    @classmethod
    def propose_code(
        cls,
        history,
        max_errors=3
    ):
        num_errors_remaining = max_errors
        while num_errors_remaining:
            print('GENERATING PROPOSAL', '=' * 30)
            history_prompt = cls.make_history_prompt(history)
            prompt = cls.prompt_prefix() + '\n' + history_prompt
            response_prefix = '<proposal name="'
            stop_sequence = '</proposal>'
            raw_proposal = prompt_llm(
                prompt,
                response_prefix=response_prefix,
                stop_sequences=[stop_sequence]
            ).strip()
            # fixup slightly bad generations
            if not raw_proposal.startswith(response_prefix):
                raw_proposal = response_prefix + raw_proposal
            if not raw_proposal.endswith(stop_sequence):
                raw_proposal += stop_sequence

            print(raw_proposal)

            proposal = cls.from_raw(raw_proposal)
            if proposal.error is not None:
                proposal.test()

            if proposal.error is not None:
                print('Error creating proposal:', proposal.error)
                history.append(proposal)
                num_errors_remaining -= 1
            else:
                # The proposal was successful
                break
        if num_errors_remaining <= 0:
            raise Exception('Too many errors!')
        return proposal

    @classmethod
    def make_history_prompt(cls, history):
        sep = '=' * 10
        prompt = [sep]
        for proposal in history:
            prompt += [
                proposal.raw,
                sep
            ]
            if proposal.error is not None:
                prompt.append('Error: ' + proposal.error)
            else:
                prompt.append(proposal.fitness_string)
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


if __name__ == '__main__':
    testcode = """
def prune_func(model: torch.nn.Module):
    for name, param in model.named_parameters():
        importance = torch.abs(param.data * param.grad)
        threshold = torch.quantile(importance, 0.2)
        mask = (torch.abs(param) > threshold).float()
        param.mul_(mask)
    """
    prune_func = get_generated_function(testcode)
    print(prune_func)

    doc = """
<proposal name="abs_mag">
<thought>
This is a simple place to start: prune the smallest weights.
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
    """.strip()
    xdoc = BeautifulSoup(doc, 'lxml')
    print(xdoc.find('code').get_text())