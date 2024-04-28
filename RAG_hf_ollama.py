import ollama
import chromadb
import pandas as pd
import chromadb.utils.embedding_functions as embedding_functions
import config

 
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=config.API['hugging_face'],
    model_name="jhgan/ko-sbert-sts"
)

client = chromadb.PersistentClient(path="DB_recycling")
collection = client.get_or_create_collection(name="recycling_chroma", embedding_function=huggingface_ef)

prompt = input('사용자 입력: ')

results = collection.query(
    query_texts=["자전거는 어떻게 버려?"],
    n_results=5
)

res = "\n".join(str(item) for item in results['documents'][0])

sys_prompt = f'''
너는 재활용 전문가로 활동하고 있어.
사람들이 재활용 방법을 물어보면 너는 주어진 Context를 바탕으로 짧게 요약해서 질문에 해당하는 내용만 답해줘야해.

Context:
{res}
'''

messages = sys_prompt + '\n' + prompt

output = ollama.generate(
  model="EEVE",
  prompt=messages
)

response_message = output['response']

print('\n[PROMPT]',sys_prompt)
print('\n[CHATBOT]\n',response_message)