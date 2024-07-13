import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from AdvancedVectorStore import AdvancedVectorStore
import os
from dotenv import load_dotenv,find_dotenv
import warnings
import wandb
import plotly.graph_objs as go


from langchain.chains import LLMChain
# Load environment variables        

load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")
os.environ['FAISS_NO_AVX2'] = '1'
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
# Initialize AdvancedVectorStore
   
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
wandb.require("core")
# Import system prompts
from system_prompts import __all__ as prompts


class AdvancedVectorStoreGUI:
    def __init__(self, master):
        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWORD")
        self.master = master
        self.master.title("Advanced VectorStore GUI")
        self.master.geometry("800x600")
        self.avs = AdvancedVectorStore(email=self.email, password=self.password)
        

        self.avs.AvancedRagChatBot.current_system_prompt
        self.create_widgets()
        
    def create_widgets(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")
        
        # Create tabs
        self.create_document_tab()
        self.create_search_tab()
        self.create_chat_tab()
        self.create_visualization_tab()
        
    def create_document_tab(self):
        doc_frame = ttk.Frame(self.notebook)
        self.notebook.add(doc_frame, text="Documents")
        
        # Add widgets for document loading and processing
        load_btn = ttk.Button(doc_frame, text="Load Documents", command=self.load_documents)
        load_btn.pack(pady=10)
        
        self.doc_text = scrolledtext.ScrolledText(doc_frame, wrap=tk.WORD, width=70, height=20)
        self.doc_text.pack(padx=10, pady=10)
        
    def create_search_tab(self):
        search_frame = ttk.Frame(self.notebook)
        self.notebook.add(search_frame, text="Search")
        
        # Add widgets for search functionality
        self.search_entry = ttk.Entry(search_frame, width=50)
        self.search_entry.pack(pady=10)
        
        search_btn = ttk.Button(search_frame, text="Search", command=self.perform_search)
        search_btn.pack()
        
        self.search_results = scrolledtext.ScrolledText(search_frame, wrap=tk.WORD, width=70, height=20)
        self.search_results.pack(padx=10, pady=10)
        
    def create_chat_tab(self):
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="Chat")
        
        # Add widgets for chat functionality
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=70, height=20)
        self.chat_history.pack(padx=10, pady=10)
        
        self.chat_entry = ttk.Entry(chat_frame, width=50)
        self.chat_entry.pack(side=tk.LEFT, padx=10)
        
        chat_btn = ttk.Button(chat_frame, text="Send", command=self.send_chat_message)
        chat_btn.pack(side=tk.LEFT)
        
    def create_visualization_tab(self):
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Add widgets for visualization
        viz_btn = ttk.Button(viz_frame, text="Generate 3D Scatterplot", command=self.generate_scatterplot)
        viz_btn.pack(pady=10)
        
    def load_documents(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                self.avs.load_documents_folder(folder_path)
                self.doc_text.insert(tk.END, f"Loaded documents from {folder_path}\n")
                self.doc_text.insert(tk.END, f"Total documents: {self.avs.document_count}\n")
                self.doc_text.insert(tk.END, f"Total chunks: {self.avs.chunk_count}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load documents: {str(e)}")
        
    def perform_search(self):
        query = self.search_entry.get()
        if query:
            results = self.avs.search(query)
            self.search_results.delete('1.0', tk.END)
            for doc in results:
                self.search_results.insert(tk.END, f"{doc.page_content}\n\n")
        
    def send_chat_message(self):
        message = self.chat_entry.get()
        if message:
            response = self.avs.chat(message)
            self.chat_history.insert(tk.END, f"You: {message}\n")
            self.chat_history.insert(tk.END, f"Bot: {response}\n\n")
            self.chat_entry.delete(0, tk.END)
        
    def generate_scatterplot(self):
        try:
            self.avs.generate_3d_scatterplot()
            messagebox.showinfo("Success", "3D Scatterplot generated and logged to wandb")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate scatterplot: {str(e)}")

if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = AdvancedVectorStoreGUI(root)
    root.mainloop()