import ollama
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import config

 
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=config.API['hugging_face'],
    model_name="BAAI/bge-m3"
)

client = chromadb.PersistentClient(path="DB_recycling")
collection = client.get_collection(name="recycling_chroma", embedding_function=huggingface_ef)

prompt = input('사용자 입력: ')

results = collection.query(
    query_texts=[prompt],
    n_results=5
)

res = "\n".join(str(item) for item in results['documents'][0])

sys_prompt = f'''
너는 재활용 전문가로 활동하고 있어.
주어진 Prompt 질문에 대한 대답을 Context를 바탕으로 짧게 요약해서 설명해줘.

Prompt:
{prompt}

Context:
{res}
'''

output = ollama.generate(
  model="Synatra",
  prompt=sys_prompt
)

response_message = output['response']

print('\n[PROMPT]',sys_prompt)
print('\n[CHATBOT]\n',response_message)