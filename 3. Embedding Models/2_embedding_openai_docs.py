<<<<<<< HEAD
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

documents =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)


## embed_query for single query

result = embedding.embed_documents(documents)

=======
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

documents =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)


## embed_query for single query

result = embedding.embed_documents(documents)

>>>>>>> ad81e3143257ccba6df184fdf2dd9b70d623ae1c
print(str(result))