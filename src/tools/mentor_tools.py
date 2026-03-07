# src/tools/mentor_tools.py
from langchain_classic.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import tool
from typing import Optional
import subprocess
import sys

class MentorTools:
    def __init__(self, retriever):
        """
        Инициализация инструментов наставника.
        
        Args:
            retriever: Retriever из векторной БД для поиска в знаниях
        """
        self.retriever = retriever
        
        # Инструмент поиска в интернете (DuckDuckGo)
        self.search_tool = DuckDuckGoSearchRun()
        
        # Инструмент выполнения Python кода
        self.python_repl_tool = PythonREPLTool()
    
    def get_knowledge_tool(self):
        """Инструмент для поиска в локальной базе знаний (RAG)"""
        
        @tool
        def search_knowledge_base(query: str) -> str:
            """
            Ищи информацию в локальной базе знаний наставника. 
            Используй для вопросов по робототехнике, компьютерному зрению, 
            тестированию и программированию.
            
            Args:
                query: Поисковый запрос на русском или английском
            """
            try:
                docs = self.retriever.invoke(query)
                if not docs:
                    return "В базе знаний нет информации по этому вопросу."
                
                # Форматируем результат
                result = "Найденная информация:\n\n"
                for i, doc in enumerate(docs, 1):
                    result += f"--- Документ {i} ---\n"
                    result += f"Содержание: {doc.page_content[:500]}...\n\n"
                
                return result
            except Exception as e:
                return f"Ошибка при поиске в базе знаний: {e}"
        
        return search_knowledge_base
    
    def get_calculator_tool(self):
        """Инструмент-калькулятор"""
        
        @tool
        def calculator(expression: str) -> str:
            """
            Вычисли математическое выражение. 
            Примеры: "2 + 2", "sin(45)", "sqrt(16)".
            
            Args:
                expression: Математическое выражение для вычисления
            """
            try:
                # Безопасное вычисление - используем только базовые математические операции
                import math
                
                # Создаем безопасное пространство имен
                safe_dict = {
                    'abs': abs, 'round': round,
                    'min': min, 'max': max,
                    'sum': sum,
                    'math': math
                }
                
                # Добавляем математические функции
                for name in dir(math):
                    if not name.startswith('_'):
                        safe_dict[name] = getattr(math, name)
                
                # Вычисляем
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                return f"Результат: {result}"
            except Exception as e:
                return f"Ошибка вычисления: {e}"
        
        return calculator  
    
    def get_safe_code_executor(self):
        """Безопасный исполнитель кода с проверкой зависимостей"""
        
        @tool
        def execute_python_code(code: str) -> str:
            """
            Выполни Python код и верни результат.
            ВАЖНО: Если код требует библиотеки вроде rospy, opencv и т.д., 
            сначала проверь их наличие.
            
            Args:
                code: Python код для выполнения
            """
            # Проверяем на опасные импорты
            dangerous_imports = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__']
            for dangerous in dangerous_imports:
                if dangerous in code:
                    return f"Код содержит потенциально опасную операцию: {dangerous}. Выполнение заблокировано."
            
            # Проверяем зависимости
            required_modules = []
            if 'import rospy' in code or 'from rospy' in code:
                required_modules.append('rospy (требует установки ROS)')
            if 'import cv2' in code:
                required_modules.append('opencv-python')
            if 'import numpy' in code:
                required_modules.append('numpy')
            
            if required_modules:
                return f"Для выполнения кода требуются модули: {', '.join(required_modules)}. Установите их командой: pip install {' '.join(required_modules)}"
            
            try:
                # Безопасное выполнение через PythonREPLTool
                result = self.python_repl_tool.invoke(code)
                return result
            except Exception as e:
                return f"Ошибка выполнения: {str(e)}"
        
        return execute_python_code
    
    def get_all_tools(self):
        """Возвращает все доступные инструменты"""
        tools = [
            Tool(
                name="Поиск в знаниях",
                func=self.get_knowledge_tool().invoke,
                description="Ищи информацию в локальной базе знаний по робототехнике, CV, тестированию и программированию."
            ),
            Tool(
                name="Поиск в интернете",
                func=self.search_tool.invoke,
                description="Ищи актуальную информацию в интернете. Используй для тем, которых нет в локальной базе знаний."
            ),
            Tool(
                name="Калькулятор",
                func=self.get_calculator_tool().invoke,
                description="Вычисляй математические выражения."
            ),
            Tool(
                name="Выполнить Python код",
                func=self.get_safe_code_executor().invoke,
                description="Выполняй Python код для демонстрации примеров, тестирования решений или вычислений. Проверяет наличие зависимостей."
            ),
        ]
        return tools