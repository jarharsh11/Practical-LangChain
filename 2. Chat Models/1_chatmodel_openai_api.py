<<<<<<< HEAD
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # Load enviornment variables from .env file

model = ChatOpenAI(model="gpt-4",
                   temperature=0.5,
                   max_completion_tokens=10)

result = model.invoke("Suggest me five best Biscuit brands in India?")

=======
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # Load enviornment variables from .env file

model = ChatOpenAI(model="gpt-4",
                   temperature=0.5,
                   max_completion_tokens=10)

result = model.invoke("Suggest me five best Biscuit brands in India?")

>>>>>>> ad81e3143257ccba6df184fdf2dd9b70d623ae1c
print(result.content)