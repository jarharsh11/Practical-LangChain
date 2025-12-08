<<<<<<< HEAD
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

=======
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

>>>>>>> ad81e3143257ccba6df184fdf2dd9b70d623ae1c
print(str(result))