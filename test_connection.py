# test_connection.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Инициализируем модель (указываем точно ту, что скачали)
llm = OllamaLLM(model="qwen2.5:7b-instruct")

# 2. Создаем промпт-шаблон
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — инженерный наставник. Отвечай кратко и по делу."),
    ("human", "{question}")
])

# 3. Создаем цепочку
chain = prompt | llm

# 4. Задаем вопрос
question = "Какие три основных компонента у ROS-ноды?"
response = chain.invoke({"question": question})

print("Вопрос:", question)
print("Ответ:", response)