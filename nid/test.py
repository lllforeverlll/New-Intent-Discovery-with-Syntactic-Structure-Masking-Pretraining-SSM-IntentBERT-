import os
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "sk-X8AYWnWOU9VXun12E8A059D235D54c758cFc04C452575774"
os.environ["OPENAI_BASE_URL"] = "https://gtapi.xiaoerchaoren.com:8932/v1"
os.environ["http_proxy"] = 'http://127.0.0.1:7890'
sentences = [
    "This is sentence 1.",
    "Another example sentence.",
    "Yet another sentence to embed."
]




client = OpenAI(
    api_key='sk-X8AYWnWOU9VXun12E8A059D235D54c758cFc04C452575774')
# 逐句获取嵌入向量
for sentence in sentences:
    response = client.embeddings.create(input=sentence, model="text-embedding-ada-002")
    embeddings.append(response['data'][0]['embedding'])


for idx, embedding in enumerate(embeddings):
    print(f"Sentence {idx + 1} embedding:", embedding)
    
    
# completion = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message.content)