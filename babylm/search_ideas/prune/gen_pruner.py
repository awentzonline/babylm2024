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
who is researching neural network model pruning algorithms.
When you respond, output an XML document "<proposal>" where the
first section ("<thought>") corresponds to your thought process when
designing the next function. The other section ("<code>")
corresponds to the exact python code that you would like to try.
Here is an example:

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
        mask = (torch.abs(param) &gt; threshold).float()
        param.mul_(mask)
</code>
</proposal>

You must use the exact function interface used above. Feel free to
define extra hyperparameters within your function as constants.
You may read from the TrainerState API to implement step-based behavior.
Keep in mind the training is going to run for 0.5 epochs or ~100 steps.
You may set `control.should_training_stop = True` if early stopping is necessary.
The function will be invoked after an SGD step is run on a batch
of input data using a causal cross entropy loss.
The model being pruned is huggingface GPT2.

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
    return '\n'.join(prompt) + '\n<proposal name="'


def get_generated_function(str_func):
    before_exec = set(locals().keys())
    exec(str_func)
    after_exec = set(locals().keys())
    new_vars = after_exec - before_exec
    for name in new_vars:
        var = locals()[name]
        if callable(var):
            return var


def propose_pruner_code(history):
    num_errors_remaining = 3
    while num_errors_remaining:
        history_prompt = make_history_prompt(history)
        prompt = PROMPT_PREFIX + '\n' + history_prompt
        print('PROMPT', '=' * 30)
        raw_proposal = prompt_llm(prompt, stop_sequences=['</proposal>'])[0].text.strip()
        if not raw_proposal.startswith('<proposal name="'):
            raw_proposal = '<proposal name="' + raw_proposal
        raw_proposal += '</proposal>'
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
