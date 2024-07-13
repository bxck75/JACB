import streamlit as st
import sys
import os
import re
import faiss
import autopep8
import coverage
import random
import requests
import numpy as np
import speech_recognition
from TTS.api import TTS
from pathlib import Path
from typing import Optional, Dict
from typing import List, Optional, Any
from hugchat import hugchat
from rich import print as pr, pretty,progress_bar,progress
from hugchat.login import Login
from system_prompts import __all__ as prompts
from langchain_community.document_loaders import PythonLoader, TextLoader, PyPDFLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever, MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from AdvancedVectorStore import AdvancedVectorStore
class DocumentLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.documents = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, directory: str) -> None:
        loaders = {
            ".py": (PythonLoader, {}),
            ".txt": (TextLoader, {}),
            ".pdf": (PyPDFLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {})
        }

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()

                if file_extension in loaders:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read()
                    except (UnicodeDecodeError, IOError):
                        st.warning(f"Skipping non-UTF-8 or unreadable file: {file_path}")
                        continue

                    loader_class, loader_args = loaders[file_extension]
                    loader = loader_class(file_path, **loader_args)
                    self.documents.extend(loader.load())

    def split_documents(self) -> None:
        splitters = {
            ".py": RecursiveCharacterTextSplitter.from_language(language="python", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            ".txt": RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            ".pdf": RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            ".html": RecursiveCharacterTextSplitter.from_language(language="html", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            ".docx": RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        }

        split_docs = []
        for doc in self.documents:
            file_extension = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
            splitter = splitters.get(file_extension, RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap))
            split_docs.extend(splitter.split_documents([doc]))

        self.documents = split_docs

class VectorStoreManager:
    def __init__(self, documents, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = self.create_indexed_vectorstore()

    def create_indexed_vectorstore(self, embedding_size=384):
        st.info("Creating indexed vectorstore...")
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore({})
        self.vectorstore = FAISS(
            self.embeddings.embed_query,
            index,
            docstore,
            {}
        )
        self.vectorstore.add_documents(self.documents)
        st.success("Indexed vectorstore created.")

    def get_basic_retriever(self, k=4):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_indexed_vectorstore() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def get_multi_query_retriever(self, k=4, llm=None):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call create_indexed_vectorstore() first.")
        return MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            llm=llm
        )

    def get_timed_retriever(self, k=1, decay_rate=1e-25):
        return TimeWeightedVectorStoreRetriever(
            vectorstore=self.vectorstore, decay_rate=decay_rate, k=k
        )

    def set_current_retriever(self, mode='basic', k=4, llm=None):
        if mode == 'multi_query':
            retriever = self.get_multi_query_retriever(k, llm)
        elif mode == 'time':
            retriever = self.get_timed_retriever(k=k)
        else:
            retriever = self.get_basic_retriever(k)
        return retriever

    def search(self, query: str, mode='basic', retriever: Optional[Any] = None, k=4, llm=None) -> List[Any]:
        if not retriever:
            retriever = self.set_current_retriever(mode=mode, k=k, llm=llm)
        return retriever.get_relevant_documents(query)

class LLMChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/', default_llm=1, default_system_prompt='default_rag_prompt'):
        self.email = email
        self.password = password
        self.current_model = 1
        self.current_system_prompt = default_system_prompt
        self.cookie_path_dir = cookie_path_dir
        self.cookies = self.login()
        self.default_llm = default_llm
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict(), default_llm=default_llm)
        self.conversation_id = None
        self.check_conv_id(self.conversation_id)

    def check_conv_id(self, id=None):
        if not self.conversation_id and not id:
            self.conversation_id = self.chatbot.new_conversation()
        else:
            if id:
                self.conversation_id = id
                self.chatbot.change_conversation(self.conversation_id)
        return self.conversation_id

    def login(self):
        st.info("Attempting to log in...")
        sign = Login(self.email, self.password)
        try:
            cookies = sign.login(cookie_dir_path=self.cookie_path_dir, save_cookies=True)
            st.success("Login successful!")
            return cookies
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.info("Attempting manual login with requests...")
            self.manual_login()
            raise

    def manual_login(self):
        login_url = "https://huggingface.co/login"
        session = requests.Session()
        response = session.get(login_url)
        st.info(f"Response Cookies: {response.cookies}")
        st.info(f"Response Content: {response.content.decode()}")

        csrf_token = response.cookies.get('csrf_token')
        if not csrf_token:
            st.error("CSRF token not found in cookies.")
            return

        login_data = {
            'email': self.email,
            'password': self.password,
            'csrf_token': csrf_token
        }

        response = session.post(login_url, data=login_data)
        if response.ok:
            st.success("Manual login successful!")
        else:
            st.error("Manual login failed!")

    def setup_speech_recognition(self):
        self.recognizer = speech_recognition.Recognizer()

    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name)

    def chat(self, message):
        return self.chatbot.chat(message)

    def query(self, message, web_search=False, stream=False, max_new_tokens=1024, temperature=0.1, top_p=0.95, repetition_penalty=1.2, top_k=50):
        return self.chatbot.query(
            text=message,
            web_search=web_search,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            truncate=1000,
            watermark=False,
            max_new_tokens=max_new_tokens,
            stop=["</s>"],
            return_full_text=False,
            stream=stream,
            _stream_yield_all=False,
            use_cache=False,
            is_retry=False,
            retry_count=5,
            conversation=None
        )
    
    def __call__(self,
            text ,
            web_search=False,
            temperature=0.1,
            top_p=20,
            repetition_penalty=1.8,
            top_k=4,
            max_new_tokens=1024,
            return_full_text=False,
            stream=False,
            stop=["</s>"],
            conversation=None
            ):
        
        return self.chatbot.query(
            text=text,
            web_search=web_search,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop=stop,
            return_full_text=return_full_text,
            stream=stream,
            conversation=conversation
        )

class CodeImprover:

    version: Optional[float] = 2.9
    script_path: Optional[str] = None
    log_file: Optional[str] = os.path.join(f"{str(Path(__file__).parent)}", "changes.txt")
    
    def __init__(self):
        self.code_marks: Dict[str, str] = {
            "python": r"```(?:python|py)\n(?P<code>[\s\S]+?)\n```",
            "javascript": r"```(?:javascript|js)\n(?P<code>[\s\S]+?)\n```",
            "java": r"```java\n(?P<code>[\s\S]+?)\n```",
            "cpp": r"```(?:cpp|c\+\+)\n(?P<code>[\s\S]+?)\n```"
        }
        self.llm = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs.logger.info("Init!")

    def read_code(self, text: str, language: str) -> Optional[str]:
        code_mark = self.code_marks.get(language)
        if code_mark:
            match = re.search(code_mark, text)
            if match:
                return match.group("code")
        return None

    def save_changes(self, text: str) -> None:
        with open(self.log_file, "a") as file:
            file.write(text)
        self.avs.logger.info(f"Changes saved to log {self.log_file}")

    def check_syntax_errors(self, code: str, language: str) -> None:
        if language == "python":
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                self.avs.logger.info(f"Syntax Error: {e}")
        # Add syntax checks for other languages if needed

    def format_code(self, code: str, language: str) -> str:
        if language == "python":
            return autopep8.fix_code(code)
        # Add code formatting for other languages if needed
        return code

    def generate_coverage_report(self, code: str, language: str) -> None:
        if language == "python":
            self.avs.logger.info(f"This is the path {str(Path(__file__).parent)}/temp.py")
            with open(f"{str(Path(__file__).parent)}/temp.py", "w") as file:
                file.write(code)
            cov = coverage.Coverage(data_file=f"{str(Path(__file__).parent)}/temp.py")
            cov.start()
            os.system(f"python {str(Path(__file__).parent)}/temp.py")
            cov.stop()
            cov.save()
            cov.report(show_missing=True, skip_covered=True, ignore_errors=True)
            os.remove(f"{str(Path(__file__).parent)}/temp.py")
            os.remove(f"{str(Path(__file__).parent)}/.coverage")

    def Enter(self, path: Optional[str] = None, language: str = "python") -> None:
        """Improves the code in a given file or all files in a given directory."""
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = '/nr_ywo/coding/JACB/test_input/' # input("[Enter the path of the file you want to improve (enter to self-improve):]")

        if path:
            if os.path.isfile(path):
                paths = [path]
            elif os.path.isdir(path):
                paths = list(Path(path).rglob("*.py")) if language == "python" else list(Path(path).rglob(f"*.{language}"))
            else:
                self.avs.logger.info(f"Invalid path.")
                return
        else:
            paths = [str(Path(__file__).parent)]
   
    def ImproveCode(self, paths):
        for path in paths:
            try:
                self.avs.logger.info(f"Paths: {path}")
                with open(path, "r") as file:
                    code = file.read()
                
                system_prompt = prompts['improver_system_prompt']
                user_prompt = prompts['improver_task_prompt']
                
                self.llm.switch_role(system_prompt=system_prompt, model_id=1)
                
                user_task_prompt = f"""User: {user_prompt}
                                       ```{language}
                                       {code}
                                       ```"""
                
                results = self.llm(user_task_prompt)
                rp(results.message_to_dict())
                if isinstance(results, list):
                    results = ''.join([str(msg) for msg in results])
                
                code = self.read_code(str(results), language)
                
                if code:
                    self.check_syntax_errors(code, language)
                    code = self.format_code(code, language)
                    self.generate_coverage_report(code, language)
                    
                    new_file_path = str(Path(path).with_name(f"{Path(path).stem}_generated_{str(self.version).replace('.', '_')}_improvement{Path(path).suffix}"))
                    
                    with open(new_file_path, "w") as file:
                        file.write(code)
                    
                    changes = results.split("I made the following changes:")
                    if len(changes) > 1:
                        changes = f"I made the following changes in version {self.version}:\n{changes[1]}"
                        self.save_changes(changes)
                    else:
                        self.avs.logger.info("No changes detected.")
                
            except FileNotFoundError as e:
                self.avs.logger.info(f"Invalid file path. {e}")


class DocumentChatbot:
    def __init__(self, llm_chatbot: LLMChatBot, vector_store_manager: VectorStoreManager):
        self.llm_chatbot = llm_chatbot
        self.vector_store_manager = vector_store_manager
        self.conversation_chain = None

    def setup_conversation_chain(self):
        retriever = self.vector_store_manager.get_basic_retriever()
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm_chatbot.chatbot,
            retriever=retriever,
            return_source_documents=True
        )

    def chat(self, query: str, chat_history: List[tuple]) -> tuple:
        if not self.conversation_chain:
            self.setup_conversation_chain()

        result = self.conversation_chain({"question": query, "chat_history": chat_history})
        return result['answer'], result['source_documents']

class StreamlitApp:
    def __init__(self):
        self.document_loader = None
        self.vector_store_manager = None
        self.llm_chatbot = None
        self.document_chatbot = None

    def run(self):
        st.title("Document Chatbot")
        self.login_and_setup(os.getenv("EMAIL"),  os.getenv("PASSWD"))
        # Sidebar for uploading documents and login
        with st.sidebar:
            st.header("Setup")
            uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "py", "html", "docx"])
            if uploaded_file:
                self.process_uploaded_file(uploaded_file)

        # Main chat interface
        if self.document_chatbot:
            st.header("Chat with your documents")
            query = st.text_input("Ask a question about your documents:")
            if query:
                with st.spinner("Thinking..."):
                    response, sources = self.document_chatbot.chat(query, [])
                st.write(response)
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source.page_content)
                        st.write(source.metadata)

    def process_uploaded_file(self, uploaded_file):
        # Save uploaded file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} has been uploaded.")

        # Load and process documents
        self.document_loader = DocumentLoader()
        self.document_loader.load_documents(".")
        self.document_loader.split_documents()

        # Create vector store
        self.vector_store_manager = VectorStoreManager(self.document_loader.documents)
        self.vector_store_manager.create_indexed_vectorstore()

    def login_and_setup(self, email, password):
        self.llm_chatbot = LLMChatBot(os.getenv("EMAIL"), os.getenv("PASSWD"))
        if self.vector_store_manager:
            self.document_chatbot = DocumentChatbot(self.llm_chatbot, self.vector_store_manager)
            st.success("Login successful and chatbot is ready!")
        else:
            st.warning("Please upload a document first.")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()

    
    llm= LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
    response = llm("Hello, How are you today?")
    rp(response)
    replicator = CodeImprover()
    replicator.improve_code(path="CodeImproverXL.py", language="python")
     
