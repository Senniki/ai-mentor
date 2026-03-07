# src/memory/vector_store.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VectorStore:
    def __init__(self, persist_directory="./chroma_db", knowledge_base_path="../knowledge_base"):
        self.persist_directory = persist_directory
        self.knowledge_base_path = knowledge_base_path
        
        # Выбираем модель для эмбеддингов (векторизации текста) - локальную!
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},  # Можно 'cuda', если есть GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Инициализируем текстовый сплиттер
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Размер чанка в символах
            chunk_overlap=200,    # Перекрытие для контекста
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_and_chunk_documents(self):
        """Загружает документы из knowledge_base и разбивает на чанки"""
        documents = []
        
        # Загружаем все .txt файлы
        txt_loader = DirectoryLoader(
            self.knowledge_base_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
        
        # Загружаем все .pdf файлы
        pdf_loader = DirectoryLoader(
            self.knowledge_base_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Можно добавить загрузку .md, .html и других форматов
        
        print(f"Загружено документов: {len(documents)}")
        
        # Разбиваем на чанки
        chunks = self.text_splitter.split_documents(documents)
        print(f"Создано чанков: {len(chunks)}")
        return chunks
    
    def create_vector_store(self, chunks):
        """Создает векторную базу данных из чанков"""
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        # vector_store.persist()
        print(f"Векторная БД создана и сохранена в {self.persist_directory}")
        return vector_store
    
    def get_retriever(self, k=4):
        """Возвращает retriever для поиска похожих чанков"""
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        return vector_store.as_retriever(search_kwargs={"k": k})