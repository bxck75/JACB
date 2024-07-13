import os
from dotenv import load_dotenv, find_dotenv
from hugchat import hugchat
from hugchat.login import Login
from rich import print as rp
from typing import List
from system_prompts import __all__ as prompts
from AdvancedVectorStore import LLMChatBot, AdvancedVectorStore
# Load environment variables
load_dotenv(find_dotenv())
email = os.getenv("EMAIL")
password = os.getenv("PASSWD")
github_token = os.getenv("GITHUB_TOKEN")

# Initialize AdvancedVectorStore with HugChat bot


class ChatBotWrapper:
    def __init__(self):
        self.bot = hugchat.ChatBot(cookies=self.cookies.get_dict(), default_llm=self.default_llm)
        self.bot.new_conversation(modelIndex=1 ,system_prompt=prompts)
        self.avs = AdvancedVectorStore(email=email, password=password)
        self.current_conversation = self.chatbot.check_conv_id()
        self.embeddings = self.avs.embeddings.embed_query()
        rp(self.avs.embeddings.model_name)
        rp(dir(self.avs ))
    def __call__(self, message: str) -> str:
        self.test()
        return self.chatbot.query(
                    message=message,
                    web_search = False,
                    stream = False,
                    use_cache = True
                )
    
    def test(self):
        rp(self.current_conversation,file='test')
        
if __name__ == '__main__':
    chatbot = ChatBotWrapper()
    while True:
        message = input("You: ")
        if message == 'exit':
            break
        
        response = chatbot(message)
        rp(f"ChatBot: {response}",file='test')

    


