from anthropic import Anthropic


client = Anthropic()


def prompt_llm(prompt, max_tokens=1024, **kwargs):
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-3-opus-20240229",
        **kwargs
    )
    return message.content