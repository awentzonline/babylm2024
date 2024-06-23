from anthropic import Anthropic


client = Anthropic()


def prompt_llm(
    prompt, response_prefix=None, max_tokens=2048, max_pages=5, **kwargs
):
    num_pages_remaining = max_pages
    response_pages = [response_prefix] if response_prefix else []
    while num_pages_remaining:
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
        if response_pages:
            prev_response = ''.join(response_pages)
            messages.append({
                "role": "assistant",
                "content": prev_response
            })

        response = client.messages.create(
            max_tokens=max_tokens,
            messages=messages,
            model="claude-3-5-sonnet-20240620",  #"claude-3-opus-20240229",  #
            **kwargs
        )

        response_text = response.content[0].text
        response_pages.append(response_text)
        if response.stop_reason != 'max_tokens':
            break
        else:
            num_pages_remaining -= 1

    return ''.join(response_pages)
