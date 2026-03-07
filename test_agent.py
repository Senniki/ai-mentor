# test_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.memory.vector_store import VectorStore
from src.tools.mentor_tools import MentorTools
from src.core.mentor_agent import MentorAgent

def test_agent():
    print("=== Тест агента-наставника (React) ===")
    
    # 1. Инициализируем векторное хранилище
    print("1. Загрузка векторной БД...")
    vs = VectorStore(
        persist_directory="./chroma_db",
        knowledge_base_path="./knowledge_base"
    )
    retriever = vs.get_retriever(k=4)
    
    # 2. Создаем инструменты
    print("2. Создание инструментов...")
    mentor_tools = MentorTools(retriever)
    tools = mentor_tools.get_all_tools()
    
    print(f"Доступно инструментов: {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
    
    # 3. Создаем агента
    print("3. Создание агента-наставника...")
    agent = MentorAgent(tools)
    
    # 4. Тестируем
    print("\n4. Тестирование агента...")
    test_queries = [
        "Составь план изучения основ компьютерного зрения на 3 занятия",
        "Объясни, что такое топики в ROS"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Запрос {i}: {query}")
        print(f"{'='*50}")
        
        response = agent.invoke(query)
        print(f"\nОтвет агента:\n{response}")
        
        # Небольшая пауза между запросами
        import time
        time.sleep(2)
    
    print(f"\n{'='*50}")
    print("Тест завершен!")
    print(f"{'='*50}")

if __name__ == "__main__":
    test_agent()