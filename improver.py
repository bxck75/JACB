import sys
import os
import re
import autopep8
import coverage
from pathlib import Path
from rich import print as rp ,print_json
from rich.console import Console
from typing import Optional, Dict
from LLMChatBot import LLMChatBot
from AdvancedVectorStore import AdvancedVectorStore, LLMChatBot
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # protected
from system_prompts import __all__ as prompts  # protected
from langchain_huggingface import HuggingFaceEndpointEmbeddings

################## Do Not Change Above Line #################### protected

'''
TODOS:
    1. Add prompt engineering
    2. Add more LLMs (maybe a local qwen0.5B model)
    3. Add more vector storages
'''

class CodeImprover:

    version: Optional[float] = 2.9
    script_path: Optional[str] = None
    log_file: Optional[str] = os.path.join(f"{str(Path(__file__).parent)}", "changes.txt")
    
    def __init__(self):
        
        self.embedder = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        self.code_marks: Dict[str, str] = {
            "python": r"```(?:python|py)\n(?P<code>[\s\S]+?)\n```",
            "javascript": r"```(?:javascript|js)\n(?P<code>[\s\S]+?)\n```",
            "java": r"```java\n(?P<code>[\s\S]+?)\n```",
            "cpp": r"```(?:cpp|c\+\+)\n(?P<code>[\s\S]+?)\n```"
        }
        
        self.llm = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.bot = self.llm.chatbot
        rp(self.bot.current_conversation)
        self.bot.get_conversation_info(self.bot.current_conversation).history
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

    def improve_code(self, path: Optional[str] = None, language: str = "python") -> None:
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
                #################################3$#
                results = self.llm(user_task_prompt)
                ####################################
                results = str(results)
                rp(results)
                if isinstance(results, list):
                    results = ''.join([str(msg) for msg in results])
                
                code = self.read_code(str(results), language)
                
                if code:
                    self.check_syntax_errors(code, language)
                    code = self.format_code(code, language)
                    self.generate_coverage_report(code, language)
                    
                    new_file_path = str(Path(path).with_name(f"{Path(path).stem}_generated_{str(self.version).replace('.', '_')}_improvement{Path(path).suffix}"))
                    Console(code, color_system='true-color', file=new_file_path, theme='dark',force_interactive=True)
                    
                    """ with open(new_file_path, "w") as file:
                        file.write(code) """
                    
                    changes = results.split("I made the following changes:")
                    if len(changes) > 1:
                        changes = f"I made the following changes in version {self.version}:\n{changes[1]}"
                        self.save_changes(changes)
                    else:
                        self.avs.logger.info("No changes detected.")
                
            except FileNotFoundError as e:
                self.avs.logger.info(f"Invalid file path. {e}")

if __name__ == "__main__":
    replicator = CodeImprover()
    replicator.improve_code(path="CodeImproverXL.py", language="python")
