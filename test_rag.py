# test_rag.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.memory.vector_store import VectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def test_rag():
    print("=== Тест RAG-конвейера ===")
    
    # 1. Инициализируем векторное хранилище
    print("1. Инициализация VectorStore...")
    vs = VectorStore(
        persist_directory="./chroma_db",
        knowledge_base_path="./knowledge_base"
    )
    
    # 2. Загружаем и чанкуем документы
    print("2. Загрузка документов...")
    chunks = vs.load_and_chunk_documents()
    
    if not chunks:
        print("ОШИБКА: Не найдены документы в knowledge_base/")
        print("Поместите файлы (.txt, .pdf) в knowledge_base/")
        return
    
    # 3. Создаем векторную БД
    print("3. Создание векторной БД...")
    vector_store = vs.create_vector_store(chunks)
    retriever = vs.get_retriever(k=3)
    
    # 4. Инициализируем модель
    print("4. Подключение к LLM...")
    llm = OllamaLLM(model="qwen2.5:7b-instruct")
    
    # 5. Создаем цепочку RAG
    print("5. Создание RAG-цепочки...\n")
    
    # Промпт для RAG
    prompt = ChatPromptTemplate.from_template("""
    Ты — инженерный наставник. Используй приведенный ниже контекст, чтобы ответить на вопрос.
    Если в контексте нет информации для ответа, скажи, что не знаешь, но можешь помочь с другими вопросами.
    Отвечай подробно и по-русски.
    
    Контекст: {context}
    
    Вопрос: {question}
    
    Ответ наставника:
    """)
    
    # RAG цепочка
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Тестируем
    test_questions = [
        "Что такое компьютерное зрение?",
        "Какие основные функции у OpenCV?",
        "Что такое фильтрация изображений?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"--- Вопрос {i}: {question}")
        try:
            answer = rag_chain.invoke(question)
            print(f"Ответ: {answer}\n")
        except Exception as e:
            print(f"Ошибка: {e}\n")
    
    print("=== Тест завершен ===")
    print("Векторная БД сохранена в папке 'chroma_db'")

if __name__ == "__main__":
    test_rag()