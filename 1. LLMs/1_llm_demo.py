<<<<<<< HEAD
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("Who won Saitama vs Garou ?")

=======
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("Who won Saitama vs Garou ?")

>>>>>>> ad81e3143257ccba6df184fdf2dd9b70d623ae1c
print(result)