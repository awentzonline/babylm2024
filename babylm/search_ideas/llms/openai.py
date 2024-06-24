import openai


client = openai.OpenAI()


def prompt_llm(
    prompt, response_prefix=None, max_tokens=2048, max_pages=5, stop_sequences=None,
    **kwargs
):
    num_pages_remaining = max_pages
    response_pages = []
    prompt += response_prefix  # TODO: openai models seem to prefer this over putting it in assistant
    while num_pages_remaining:
        messages = [
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

        response = client.chat.completions.create(
            model="gpt-4o",  # "gpt-3.5-turbo",  # "gpt-4",
            max_tokens=max_tokens,
            messages=messages,
            stop=stop_sequences,
            **kwargs
        )
        response_text = response.choices[0].message.content
        response_pages.append(response_text)
        if response.choices[0].finish_reason != 'length':
            break
        else:
            num_pages_remaining -= 1

    return ''.join(response_pages)
