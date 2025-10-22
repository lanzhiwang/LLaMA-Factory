from openai import OpenAI

client = OpenAI(
    api_key="0",
    base_url="http://0.0.0.0:7860/v1"
)

messages = [{"role": "user", "content": "Who are you?"}]
result = client.chat.completions.create(messages=messages, model="/root/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct")
print(result.choices[0].message)
