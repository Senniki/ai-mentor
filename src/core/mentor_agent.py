# src/core/mentor_agent.py
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.agents import Tool
from langchain_ollama import OllamaLLM
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from typing import List
import warnings
import re

warnings.filterwarnings('ignore')

class MentorAgent:
    def __init__(self, tools: List[Tool], model_name="qwen2.5:7b-instruct"):
        """
        Инициализация агента-наставника (React агент).
        """
        self.model_name = model_name
        
        # Инициализируем LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3,
            num_predict=2048
        )
        
        # Промпт для React-агента
        react_prompt = PromptTemplate.from_template(
            """Ты — инженерный наставник по робототехнике, компьютерному зрению, тестированию и программированию.

            У тебя есть доступ к следующим инструментам:
            {tools}

            Используй следующий формат СТРОГО:

            Question: вопрос пользователя
            Thought: подумай, что нужно сделать
            Action: название инструмента из [{tool_names}]
            Action Input: входные данные для инструмента
            Observation: результат выполнения инструмента
            ... (повторяй Thought/Action/Action Input/Observation если нужно)
            Thought: я получил достаточно информации
            Final Answer: окончательный ответ на русском языке

            ВАЖНЫЕ ПРАВИЛА:
            1. Всегда используй ТОЛЬКО инструменты из списка
            2. После каждого Action должно быть Action Input
            3. После Action Input всегда идет Observation
            4. Не используй Markdown или форматирование
            5. Отвечай на русском языке

            Начинаем!

            Question: {input}
            Thought: {agent_scratchpad}
            """
        )
        
        # Создаем React-агента
        react_agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=react_prompt
        )
        
        # Создаем AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="force",
            return_intermediate_steps=True
        )
        
        # Память для истории
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chat_history = []
    
    def invoke(self, query: str) -> str:
        """
        Выполняет запрос через агента.
        """
        try:
            # Добавляем в историю
            self.chat_history.append({"role": "user", "content": query})
            
            # Формируем историю для контекста
            history_text = ""
            for msg in self.chat_history[-6:-1]:  # Последние 3 сообщения (кроме текущего)
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            
            # Запускаем агента
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": history_text
            })
            
            # Извлекаем ответ
            output = response.get("output", str(response))
            
            # Очищаем ответ от возможных артефактов
            output = self._clean_response(output)
            
            # Сохраняем ответ
            self.chat_history.append({"role": "assistant", "content": output})
            
            return output
            
        except Exception as e:
            error_msg = str(e)
            print(f"Детали ошибки: {error_msg}")
            
            # Если ошибка парсинга, пробуем извлечь ответ из промежуточных шагов
            if "Could not parse LLM output" in error_msg:
                # Пытаемся найти Final Answer в тексте ошибки
                match = re.search(r"Final Answer:?(.*)", error_msg, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # Возвращаем дружелюбное сообщение
            return "Извините, я не смог правильно обработать запрос. Попробуйте переформулировать вопрос более конкретно."
    
    def _clean_response(self, text: str) -> str:
        """Очищает ответ от артефактов"""
        # Удаляем возможные остатки формата React
        text = re.sub(r'Thought:.*?(?=Final Answer:|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'Action:.*?(?=Final Answer:|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'Action Input:.*?(?=Final Answer:|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'Observation:.*?(?=Final Answer:|$)', '', text, flags=re.DOTALL)
        
        # Извлекаем Final Answer если есть
        match = re.search(r'Final Answer:?(.*)', text, re.IGNORECASE)
        if match:
            text = match.group(1)
        
        return text.strip()
    
    def clear_memory(self):
        """Очищает историю"""
        self.chat_history = []
        self.memory.clear()