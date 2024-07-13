import streamlit as st
import os
from typing import List, Dict
import uuid
from system_prompts import __all__ as prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, GPT2TokenizerFast
from langchain_huggingface import HuggingFacePipeline
from FaissStorage import AdvancedVectorStore

class Conversation:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.messages: List[Dict[str, str]] = []

class StreamlitRAGApp:
    def __init__(self):
        self.vector_store = AdvancedVectorStore(
            email=os.getenv("EMAIL"),
            password=os.getenv("PASSWD")
        )
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation_id: str = None
        self.init_session_state()

    def init_session_state(self):
        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if 'current_conversation_id' not in st.session_state:
            st.session_state.current_conversation_id = None

    def create_new_conversation(self):
        conv_id = str(uuid.uuid4())
        conv_name = f"Conversation {len(self.conversations) + 1}"
        new_conv = Conversation(conv_id, conv_name)
        self.conversations[conv_id] = new_conv
        st.session_state.conversations[conv_id] = new_conv
        self.current_conversation_id = conv_id
        st.session_state.current_conversation_id = conv_id

    def delete_conversation(self, conv_id: str):
        del self.conversations[conv_id]
        del st.session_state.conversations[conv_id]
        if self.current_conversation_id == conv_id:
            self.current_conversation_id = None
            st.session_state.current_conversation_id = None

    def switch_conversation(self, conv_id: str):
        self.current_conversation_id = conv_id
        st.session_state.current_conversation_id = conv_id

    def add_message(self, role: str, content: str):
        if self.current_conversation_id:
            self.conversations[self.current_conversation_id].messages.append({"role": role, "content": content})

    def render_sidebar(self):
        st.sidebar.title("Conversations")
        
        if st.sidebar.button("New Conversation"):
            self.create_new_conversation()

        for conv_id, conv in self.conversations.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(conv.name, key=f"conv_{conv_id}"):
                    self.switch_conversation(conv_id)
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    self.delete_conversation(conv_id)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Settings")

        # VectorStore selector
        vector_store_options = ["FAISS", "Pinecone", "Milvus"]
        selected_vs = st.sidebar.selectbox("Vector Store", vector_store_options)

        # Retriever selector
        retriever_options = ["Basic", "Compressed", "Self-Query", "Multi-Query", "Time-Weighted"]
        selected_retriever = st.sidebar.selectbox("Retriever", retriever_options)

        # System prompt selector
        system_prompt_options = ["default_rag_prompt", "copilot_prompt", "custom"]
        selected_prompt = st.sidebar.selectbox("System Prompt", system_prompt_options)

        if selected_prompt == "custom":
            custom_prompt = st.sidebar.text_area("Custom Prompt")

        # Web search toggle
        web_search_enabled = st.sidebar.toggle("Enable Web Search")

        # Assistant selector
        assistant_options = ["Claude", "GPT-3.5", "GPT-4"]
        selected_assistant = st.sidebar.selectbox("Assistant", assistant_options)

        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            k_value = st.number_input("K value for retriever", min_value=1, max_value=20, value=4)
            similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.78, step=0.01)

    def render_chat_interface(self):
        st.title("RAG Chat Interface")

        if not self.current_conversation_id:
            st.warning("Please create or select a conversation to start chatting.")
            return

        for message in self.conversations[self.current_conversation_id].messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is your question?"):
            self.add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = self.vector_store.rag_chat(prompt)
                st.markdown(response)
                self.add_message("assistant", response)

    def run(self):
        self.render_sidebar()
        self.render_chat_interface()

if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()