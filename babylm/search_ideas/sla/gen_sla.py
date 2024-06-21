# Based on https://arxiv.org/pdf/2406.08414
from dataclasses import dataclass
import json
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import torch
import transformers
import numpy as np

from babylm.search_ideas.llms.anthropic import prompt_llm


PROMPT_PREFIX = """"
You're a machine learning engineer, mathematician, and evolutionary neurobiologist
who is researching sub-quadratic causal self-attention algorithms for transformer models.

When you respond, output an XML document "<proposal>" where the
first section ("<thought>") corresponds to your thought process when
designing the next function. The other section ("<code>")
corresponds to the exact python code that you would like to try.
Here is an example:

<proposal name="mult_cumsum">
<thought>
This is a simple place to start.
</thought>
<code>
def mult_cumsum_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
):
    kv = keys * values  # bind keys and values
    kvt = kv.cumsum(dim=1)  # cumsum over sequence is causal
    return kvt * queries  # retrieve queried values at each step
</code>
</proposal>

You must use the exact function interface used above. Feel free to
define extra hyperparameters within your function as constants.

Important:
 * The input tensors all have shape (batch, sequence, num_heads, head_dims)
 * Make sure your self-attention algorithm is both causal and sub-quadratic.

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
        return proposal


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
def abs_mag_prune(
    model: torch.nn.Module,
    state: transformers.TrainerState,
    control: transformers.TrainerControl
):
    for name, param in model.named_parameters():
        importance = torch.abs(param.data * param.grad)
        threshold = torch.quantile(importance, 0.2)
        mask = (2 < 4 or torch.abs(param) > threshold).float()
        param.mul_(mask)
</code>
</proposal>
    """.strip()
    xdoc = BeautifulSoup(doc, 'lxml')
    print(xdoc.find('code').get_text())
