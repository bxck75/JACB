
import sys
import os
import re
import autopep8
import coverage
import random
import numpy as np
from pathlib import Path
from typing import Optional
from LLMChatBot import LLMChatBot
from FaissStorage import AdvancedVectorStore
sys.path.append(str(Path(__file__).parent.parent.parent.parent)) # protected
from system_prompts import __all__ as prompts                   # protected
################## Do Not Change Above Line #################### protected

# TODOS list goes here we need more effective improvment 
# 1. Add prompt enginering 
# 2. Add more LLMs (maybe a local qwen0.5B model)
# 3. Add more vector storages


class CodeImprover:

    version: Optional[float] = 2.8
    script_path: Optional[str] = None
    log_file: Optional[str] = os.path.join(f"{str(Path(__file__).parent)}","changes.txt")
    
    def __init__(self):
        self.python_mark: Optional[str] = r"```(python|py|)\n(?P<code>[\s\S]+?)\n```"
        self.llm = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs.logger.info("Init!")

    @staticmethod
    def read_code(text: str, code_mark: str) -> str:
        match = re.search(code_mark, text)
        if match:
            return match.group("code")

    def save_changes(self, text: str) -> None:
        with open(self.log_file, "a") as file:
            file.write(text)
        self.avs.logger.info(f"Changes saved to log {self.log_file}")
    
    # Debug Tools
    def check_syntax_errors(self, code: str) -> None:
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            self.avs.logger.info(f"Syntax Error: {e}")

    def format_code(self, code: str) -> str:
        return autopep8.fix_code(code)

    def generate_coverage_report(self, code: str) -> None:
        self.avs.logger.info(f"this is the path {str(Path(__file__).parent)}/temp.py")
        with open(f"{str(Path(__file__).parent)}/temp.py", "w") as file:
            file.write(code)
        cov = coverage.Coverage(data_file=f"{str(Path(__file__).parent)}/temp.py")
        cov.start()
        os.system(f"python {str(Path(__file__).parent)}/temp.py")
        cov.stop()
        cov.save()
        cov.report(show_missing=True,skip_covered=True,ignore_errors=True)
        os.remove(f"{str(Path(__file__).parent)}/temp.py")
        os.remove(f"{str(Path(__file__).parent)}/.coverage")

    def improve_code(self, path: Optional[str] = None) -> None:
        """Improves the code in a given file or all files in a given directory."""
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = input("[Enter the path of the file you want to improve(enter to self-improve):]")
        if path != "": # not empty
            if os.path.isfile(path):
                paths = [path]
            elif os.path.isdir(path): # path
                paths = list(Path(path).rglob("*.py")) # glob all python file
            else:
                self.avs.logger.info(f"Invalid path.")
                return
        else: # if all else fails...self improve!
            paths = [str(Path(__file__).parent)]

        # For each path in the list
        for path in paths:
            try: # Lets fetch the content 
                self.avs.logger.info(f"Paths: {path}")
                with open(path, "r") as file:
                    code = file.read()
                responses = [] # list for the responses
                ###############################################
                # Do Not Change Below This Line #  CORE!!!   #
                ###############################################
                system_prompt=prompts['improver_system_prompt']
                user_prompt=prompts['improver_task_prompt']
                # set the system prompt for the  bot
                self.llm.switch_role(
                    system_prompt=system_prompt, model_id=1)
                #self.avs.logger.info(system_prompt)
                # set the user prompt for the  bot
                user_task_prompt= f"""User :{user_prompt}
                                        ```py
                                        {code}
                                        ```
                                    """

                results = self.llm(user_task_prompt) #  IN/OUT #
                #self.avs.logger.info(f'[Result!:{results}]') # debug
                #for chunk in results: # Here we run the llm!
                #    responses.append(chunk)
                    #self.avs.logger.info(f'Chunk!{chunk}') # debug
                self.avs.logger.info(f'[Final!:{results}]') # debug
                #read detect the code in the output
                code = self.read_code(results, self.python_mark)
                ################################################
                # Do Not Change Above This Line #  CORE!!!     #
                ################################################
                # Debugging
                self.avs.logger.info(f'[Debugging!:{responses}]') # debug
                if code:
                    # Check for syntax errors
                    self.check_syntax_errors(code)
                    # Format the code
                    code = self.format_code(code)
                    # Generate a code coverage report
                    self.generate_coverage_report(code)

                    #self.avs.logger.info(f'[Debugging!:{responses}]') # debug
                    # Extract Name and Save code 
                    new_file_path = str(Path(path).with_name(
                        f"{Path(path).stem}_generated_{str(self.version).replace('.', '_')}_improvement{Path(path).suffix}"))
                    # write
                    with open(new_file_path, "w") as file:
                        file.write(code)
                    #self.avs.logger.info(f"[Improved code saved to {new_file_path}]")

                # split off the changes and save them
                changes = responses.split("I made the following changes:")
                changes = f"I made the following changes in version {self.version}:\n{changes.pop()}"
                self.save_changes(changes)

            except FileNotFoundError as e:
                self.avs.logger.info(f"Invalid file path.{e}")

if __name__ == "__main__":
    Replicator = CodeImprover()
    Replicator.improve_code(path="CodeImproverXL.py")