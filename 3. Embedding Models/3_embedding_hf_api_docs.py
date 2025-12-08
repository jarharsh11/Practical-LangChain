<<<<<<< HEAD
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction")

texts =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

result = embedding.embed_documents(texts)

=======
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction")

texts =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

result = embedding.embed_documents(texts)

>>>>>>> ad81e3143257ccba6df184fdf2dd9b70d623ae1c
print(str(result))