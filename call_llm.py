from openai import OpenAI


def query_gpt(prompt, llm_api_key, model_name="gpt-4o-mini", temperature=1):

    client = OpenAI(api_key=llm_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model_name,
        temperature=temperature
    )
    return chat_completion.choices[0].message.content
