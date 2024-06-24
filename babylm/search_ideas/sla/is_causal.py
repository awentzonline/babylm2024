from babylm.search_ideas.llms.openai import prompt_llm


PROMPT = """
You're a machine learning researcher exploring causal self-attention algorithms.
Given a function, evaluate whether it leaks information to future tokens thereby
violating the causal constraint.

```
%s
```

Respond as follows: If there is no violation output "CAUSAL.", else output "NOT CAUSAL."
and the specific code which causes the problem. No more and no less.
""".strip()


def check_is_causal(code):
    prompt = PROMPT % code
    response = prompt_llm(prompt).strip()
    if response.lower().startswith('causal'):
        return True, response
    else:
        return False, response
