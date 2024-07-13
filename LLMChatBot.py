
import os
import tempfile
from datetime import datetime
import webbrowser
from tkinter import Toplevel
import warnings
import faiss,logging
import numpy as np
import wandb
from typing import List, Dict, Any, Optional, Union
from git import Repo
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import requests
from rich import print as rp
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from dotenv import load_dotenv, find_dotenv
import speech_recognition
from TTS.api import TTS
from sklearn.decomposition import PCA
from playsound import playsound
from hugchat import hugchat
from hugchat.login import Login
import plotly.graph_objs as go
from langchain_core.documents import Document
from langchain_community.llms.huggingface_text_gen_inference import (HuggingFaceTextGenInference)
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.llms.self_hosted_hugging_face import SelfHostedHuggingFaceLLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    PythonLoader
)

# Load environment variables
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ['FAISS_NO_AVX2'] = '1'
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Import system prompts
from system_prompts import __all__ as prompts

class LLMChatBot:
    def __init__(self, email, password, cookie_path_dir='./cookies/', default_llm=1, default_system_prompt='default_rag_prompt'):
        self.email = email
        self.password = password
        self.current_model = 1
        self.current_system_prompt=default_system_prompt
        self.cookie_path_dir = cookie_path_dir
        self.cookies = self.login()
        self.default_llm = default_llm
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict(), default_llm=default_llm,system_prompt=prompts[default_system_prompt])
        self.conversation_id=None
        self.check_conv_id(self.conversation_id)
        rp("[self.conversation_id:{self.conversation_id}]")

    def check_conv_id(self, id=None):
        if not self.conversation_id and not id:
            self.conversation_id = self.chatbot.new_conversation(modelIndex=self.current_model,system_prompt=self.current_system_prompt)
        else:
            if id:
                self.conversation_id=id
                self.chatbot.change_conversation(self.conversation_id)
        
        return self.conversation_id        

    def login(self):
        rp("Attempting to log in...")
        sign = Login(self.email, self.password)
        try:
            cookies = sign.login(cookie_dir_path=self.cookie_path_dir, save_cookies=True)
            rp("Login successful!")
            return cookies
        except Exception as e:
            rp(f"Login failed: {e}")
            rp("Attempting manual login with requests...")
            self.manual_login()
            raise

    def manual_login(self):
        login_url = "https://huggingface.co/login"
        session = requests.Session()
        response = session.get(login_url)
        rp("Response Cookies:", response.cookies)
        rp("Response Content:", response.content.decode())
        
        csrf_token = response.cookies.get('csrf_token')
        if not csrf_token:
            rp("CSRF token not found in cookies.")
            return
        
        login_data = {
            'email': self.email,
            'password': self.password,
            'csrf_token': csrf_token
        }
        
        response = session.post(login_url, data=login_data)
        if response.ok:        

            rp("Manual login successful!")
        else:
            rp("Manual login failed!")

    def setup_speech_recognition(self):
        self.recognizer = speech_recognition.Recognizer()
    
    def setup_tts(self, model_name="tts_models/en/ljspeech/fast_pitch"):
        self.tts = TTS(model_name=model_name)

    def chat(self, message):
        return self.chatbot.chat(message)
    
    def query(self,
                message, 
                web_search=False, 
                stream=False,
                max_new_tokens = 1024, 
                temperature = 0.1,
                top_p = 0.95,
                repetition_penalty = 1.2,
                top_k = 50
            ):
        return self.chatbot.query(
            text=message,
            web_search = web_search,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            top_k = top_k,
            truncate = 1000,
            watermark = False,
            max_new_tokens = max_new_tokens,
            stop = ["</s>"],
            return_full_text = False,
            stream = stream,
            _stream_yield_all = False,
            use_cache = False,
            is_retry = False,
            retry_count = 5,
            conversation = None
        )
    
    def stream_response(self, message):
        for resp in self.query(message, stream=True):
            rp(resp)

    def web_search(self, query):
        query_result = self.query(query, web_search=True)
        results = []
        for source in query_result.web_search_sources:
            results.append({
                'link': source.link,
                'title': source.title,
                'hostname': source.hostname
            })
        return results

    def create_new_conversation(self, switch_to=True):
        return self.chatbot.new_conversation(switch_to=switch_to, modelIndex=self.current_model, system_prompt=self.current_system_prompt)

    def get_remote_conversations(self):
        return self.chatbot.get_remote_conversations(replace_conversation_list=True)

    def get_local_conversations(self):
        return self.chatbot.get_conversation_list()

    def get_available_models(self):
        return self.chatbot.get_available_llm_models()

    def switch_model(self, index):
        self.chatbot.switch_llm(index)

    def switch_conversation(self, id):
        self.conv_id = id
        self.chatbot.change_conversation(self.conv_id)

    def get_assistants(self):
        return self.chatbot.get_assistant_list_by_page(1)

    def switch_role(self, system_prompt, model_id=1):
        self.chatbot.delete_all_conversations()
        self.check_conv_id = self.chatbot.new_conversation(switch_to=True, system_prompt=system_prompt,  modelIndex=model_id)
        return self.check_conv_id
    
    def __run__(self, message):
        if not self.conversation_id:
            self.conversation_id = self.chatbot.new_conversation(modelIndex=self.current_model,
                                                                 system_prompt=self.current_system_prompt,
                                                                 switch_to=True)
        return self.query(message)
    
    def __call__(self, message):
        if not self.conversation_id:
            self.conversation_id = self.chatbot.new_conversation(modelIndex=self.current_model,
                                                                 system_prompt=self.current_system_prompt,
                                                                 switch_to=True)
        return self.chat(message)



if __name__ == "__main__":
    llm= LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
    response = llm("Hello, How are you today?")
    rp(response)