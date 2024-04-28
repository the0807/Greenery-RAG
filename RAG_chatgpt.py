import openai
import chromadb
from chromadb.utils import embedding_functions
import config


def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input = text)
    print(response["data"][0]["embedding"])
    return response["data"][0]["embedding"]
 
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key = config.API['chatgpt'],
                model_name = "text-embedding-ada-002"
            )

client = chromadb.PersistentClient(path="DB")
collection = client.get_collection("recycling",embedding_function=openai_ef)

prompt = input('사용자 입력: ')

vector = text_embedding(prompt)

results = collection.query(    
    query_embeddings = vector,
    n_results = 5,
    include = ["documents"]
)

res = "\n".join(str(item) for item in results['documents'][0])

sys_prompt = f'''
너는 재활용 전문가로 활동하고 있어.
사람들이 재활용 방법을 물어보면 너는 주어진 Context를 바탕으로 짧게 요약해서 중요한 정보만 답해줘야해.

Context:
{res}
'''

messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.1
)
response_message = response["choices"][0]["message"]["content"]

print('\n[PROMPT]',sys_prompt)
print('\n[CHATBOT]\n',response_message)