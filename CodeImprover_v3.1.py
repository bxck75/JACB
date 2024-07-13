import sys
import os
import re
import autopep8
import coverage
import random
import numpy as np
import subprocess
from hugchat import hugchat
from pathlib import Path
from typing import Optional
from rich import print as rp
from LLMChatBot import LLMChatBot
from FaissStorage import AdvancedVectorStore
from typing import List, Dict, Any, Optional, Union
sys.path.append(str(Path(__file__).parent.parent.parent.parent)) # protected
from system_prompts import __all__ as prompts                   # protected
################## Do Not Change Above Line #################### protected

'''
TODOS:

    -Add Real-time code execution , monitoring error output...fix and itterate so improvement goes faster
'''

class CodeImprover:
    version: float = 3.0
    script_path: Optional[str] = None
    log_file: str = os.path.join(str(Path(__file__).parent), "changes.txt")
    
    def __init__(self):
        self.code_marks: Dict[str, str] = {
            "python": r"```(?:python|py)\n(?P<code>[\s\S]+?)\n```",
            "javascript": r"```(?:javascript|js)\n(?P<code>[\s\S]+?)\n```",
            "java": r"```java\n(?P<code>[\s\S]+?)\n```",
            "cpp": r"```(?:cpp|c\+\+)\n(?P<code>[\s\S]+?)\n```"
        }

        self.llm = LLMChatBot(
            email=os.getenv("EMAIL"), 
            password=os.getenv("PASSWD"),
            default_llm = 1,
            default_system_prompt = 'default_rag_prompt')
        
       
        self.avs = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs.logger.info("CodeImprover initialized")

    
    def read_code(text: str, code_mark: str) -> Optional[str]:
        rp(text)
        match = re.search(code_mark, text)
        return match.group("code") if match else None

    def save_changes(self, text: str) -> None:
        with open(self.log_file, "a") as file:
            file.write(text)
        self.avs.logger.info(f"Changes saved to log {self.log_file}")
    
    def check_syntax_errors(self, code: str, language: str) -> List[str]:
        errors = []
        if language == "python":
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                errors.append(f"Syntax Error: {e}")
        elif language in ["javascript", "java", "cpp"]:
            # For other languages, we'll use a simple regex-based check for now
            # This is not comprehensive and should be replaced with proper parsers
            unmatched_brackets = re.findall(r'[{[(]}])', code)
            if unmatched_brackets:
                errors.append(f"Possible unmatched brackets: {unmatched_brackets}")
        return errors

    def format_code(self, code: str, language: str) -> str:
        if language == "python":
            return autopep8.fix_code(code)
        # For other languages, we'll return the code as-is for now
        # TODO: Integrate formatters for other languages
        return code

    def generate_coverage_report(self, code: str, language: str) -> None:
        if language != "python":
            self.avs.logger.info(f"Coverage report not supported for {language}")
            return
        
        temp_file = f"{str(Path(__file__).parent)}/temp.py"
        with open(temp_file, "w") as file:
            file.write(code)
        
        try:
            cov = coverage.Coverage(data_file=temp_file)
            cov.start()
            subprocess.run(["python", temp_file], check=True, capture_output=True, text=True, timeout=5)
            cov.stop()
            cov.save()
            cov.report(show_missing=True, skip_covered=True, ignore_errors=True)
        except subprocess.TimeoutExpired:
            self.avs.logger.warning("Code execution timed out")
        except subprocess.CalledProcessError as e:
            self.avs.logger.error(f"Error executing code: {e.stderr}")
        finally:
            os.remove(temp_file)
            if os.path.exists(f"{str(Path(__file__).parent)}/.coverage"):
                os.remove(f"{str(Path(__file__).parent)}/.coverage")

    def improve_code(self, path: Optional[str] = None) -> None:
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = input("[Enter the path of the file you want to improve (enter to self-improve):] ")
        
        if path:
            if os.path.isfile(path):
                paths = [path]
            elif os.path.isdir(path):
                paths = list(Path(path).rglob("*"))  # glob all files
            else:
                self.avs.logger.error(f"Invalid path: {path}")
                return
        else:
            paths = [str(Path(__file__))]

        for path in paths:
            try:
                self.avs.logger.info(f"Processing: {path}")
                with open(path, "r") as file:
                    code = file.read()
                
                language = path.suffix[1:]  # Get language from file extension
                if language not in self.code_marks:
                    self.avs.logger.warning(f"Unsupported language: {language}. Skipping file.")
                    continue

                system_prompt = prompts['improver_system_prompt']
                user_prompt = prompts['improver_task_prompt']
                self.llm.switch_role(system_prompt=system_prompt, model_id=1)
                
                user_task_prompt = f"""User: {user_prompt}
                                    ```{language}
                                    {code}
                                    ```
                                """

                results = self.llm(user_task_prompt)
                self.avs.logger.info(f'LLM Response received')
                
                improved_code = self.read_code(results, self.code_marks[language])
                if improved_code:
                    errors = self.check_syntax_errors(improved_code, language)
                    for error in errors:
                        self.avs.logger.warning(f"Syntax error detected: {error}")
                    
                    formatted_code = self.format_code(improved_code, language)
                    
                    if language == "python":
                        self.generate_coverage_report(formatted_code, language)

                    new_file_path = str(Path(path).with_name(f"{Path(path).stem}_improved_v{self.version}{Path(path).suffix}"))
                    with open(new_file_path, "w") as file:
                        file.write(formatted_code)
                    self.avs.logger.info(f"Improved code saved to {new_file_path}")

                    changes = results.split("I made the following changes:")
                    changes = f"Changes made in version {self.version}:\n{changes[-1] if len(changes) > 1 else 'No specific changes reported.'}"
                    self.save_changes(changes)
                else:
                    self.avs.logger.warning("No improved code found in LLM response")

            except FileNotFoundError:
                self.avs.logger.error(f"File not found: {path}")
            except Exception as e:
                self.avs.logger.error(f"Error processing {path}: {str(e)}")

if __name__ == "__main__":
    improver = CodeImprover()
    improver.improve_code()
