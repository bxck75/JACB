import tkinter as tk
from tkinter import ttk

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

import tkinter.scrolledtext as scrolledtext
from ttkthemes import ThemedTk
import csv
from datetime import datetime
from FaissStorage import AdvancedVectorStore
class ChatbotApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="equilux")
        self.title("Chatbot Application")
        self.attributes('-fullscreen', True)
        self.system_prompts = prompts
        self.system_prompt = prompts['default_rag_prompt']
        self.StoreManager = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.StoreManager.set_bot_role()
        self.style = ttk.Style(self)
        self.style.configure('TNotebook', background='#2e2e2e')
        self.style.configure('TNotebook.Tab', background='#2e2e2e', foreground='white')
        self.style.map('TNotebook.Tab', background=[('selected', '#3e3e3e')])
        
        self.create_widgets()
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create tabs
        self.create_vectorstore_tab()
        self.create_websearch_tab()
        self.create_vectorstore_manager_tab()
        
        # Exit button
        exit_button = ttk.Button(self, text="Exit", command=self.quit)
        exit_button.pack(pady=10)
    
    def create_vectorstore_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='VectorStore Chatbot')
        mode='basic'
        k=5
        similarity_threshold=0.75
        retriever = self.set_current_retriever(mode=mode, k=k, sim_rate=similarity_threshold)
        # Chat display
        self.vs_chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, bg='#1e1e1e', fg='white')
        self.vs_chat_display.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Input field
        self.vs_input = ttk.Entry(tab, width=50)
        self.vs_input.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Send button
        send_button = ttk.Button(tab, text="Send", command=self.vs_send_message)
        send_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Feedback frame
        feedback_frame = ttk.Frame(tab)
        feedback_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Star rating
        self.vs_rating = tk.IntVar()
        for i in range(1, 6):
            ttk.Radiobutton(feedback_frame, text=f"{i} ★", variable=self.vs_rating, value=i).pack(side=tk.LEFT)
        
        # Submit feedback button
        submit_feedback = ttk.Button(feedback_frame, text="Submit Feedback", command=self.vs_submit_feedback)
        submit_feedback.pack(side=tk.LEFT, padx=10)
    
    def create_websearch_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Web Search Chatbot')
        
        # Chat display
        self.ws_chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, bg='#1e1e1e', fg='white')
        self.ws_chat_display.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Input field
        self.ws_input = ttk.Entry(tab, width=50)
        self.ws_input.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Send button
        send_button = ttk.Button(tab, text="Send", command=self.ws_send_message)
        send_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Feedback frame
        feedback_frame = ttk.Frame(tab)
        feedback_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Star rating
        self.ws_rating = tk.IntVar()
        for i in range(1, 6):
            ttk.Radiobutton(feedback_frame, text=f"{i} ★", variable=self.ws_rating, value=i).pack(side=tk.LEFT)
        
        # Submit feedback button
        submit_feedback = ttk.Button(feedback_frame, text="Submit Feedback", command=self.ws_submit_feedback)
        submit_feedback.pack(side=tk.LEFT, padx=10)
    
    def create_vectorstore_manager_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='VectorStore Manager')
        
        # Add your VectorStore management tools here
        label = ttk.Label(tab, text="VectorStore Management Tools")
        label.pack(padx=10, pady=10)
    
    def vs_send_message(self):
        message = self.vs_input.get()
        self.vs_chat_display.insert(tk.END, f"You: {message}\n")
        
        # Here you would integrate with your VectorStoreToolkit chatbot
        response = "This is a placeholder response from the VectorStore chatbot."
        
        self.vs_chat_display.insert(tk.END, f"Assistant: {response}\n\n")
        self.vs_input.delete(0, tk.END)
    
    def ws_send_message(self):
        message = self.ws_input.get()
        self.ws_chat_display.insert(tk.END, f"You: {message}\n")
        
        # Here you would integrate with your web search chatbot
        response = "This is a placeholder response from the web search chatbot."
        
        self.ws_chat_display.insert(tk.END, f"Assistant: {response}\n\n")
        self.ws_input.delete(0, tk.END)
    
    def vs_submit_feedback(self):
        rating = self.vs_rating.get()
        last_message = self.vs_chat_display.get("end-2l linestart", "end-1l lineend")
        user_message = self.vs_chat_display.get("end-4l linestart", "end-3l lineend").replace("You: ", "")
        self.save_feedback("vectorstore", user_message, last_message, rating)
        self.vs_rating.set(0)
    
    def ws_submit_feedback(self):
        rating = self.ws_rating.get()
        last_message = self.ws_chat_display.get("end-2l linestart", "end-1l lineend")
        user_message = self.ws_chat_display.get("end-4l linestart", "end-3l lineend").replace("You: ", "")
        self.save_feedback("websearch", user_message, last_message, rating)
        self.ws_rating.set(0)
    
    def save_feedback(self, chatbot_type, user_message, assistant_message, rating):
        with open('dataset.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), chatbot_type, user_message, assistant_message, rating])

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()