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
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # protected
from system_prompts import __all__ as prompts  # protected

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
    log_file: Optional[str] = os.path.join(f"{str(Path(__file__).parent)}","changes.txt")
    
    def __init__(self):
        self.python_mark: Optional[str] = r"```(python|py|)\n(?P<code>[\s\S]+?)\n```"
        self.llm = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs.logger.info("Init!")

    @staticmethod
    def read_code(text: str, code_mark: str) -> Optional[str]:
        if isinstance(text, str):
            match = re.search(code_mark, text)
            if match:
                return match.group("code")
        return None

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
            path = input("[Enter the path of the file you want to improve (enter to self-improve):]")
        if path!= "": # not empty
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
                # Ensure results is a string
                results_str = str(results)
                #read detect the code in the output
                code = self.read_code(results_str, self.python_mark)
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
                changes = results_str.split("I made the following changes:")
                changes = f"I made the following changes in version {self.version}:\n{changes.pop()}"
                self.save_changes(changes)

            except FileNotFoundError as e:
                self.avs.logger.info(f"Invalid file path.{e}")

if __name__ == "__main__":
    replicator = CodeImprover()
    replicator.improve_code(path="CodeImproverXL.py")

'''
Description:
    This script is a code improver that takes a Python file or directory as input and improves the code by checking for syntax errors, formatting the code, and generating a code coverage report.

Usage:
    The script can be used by creating an instance of the CodeImprover class and calling the improve_code method with the path of the file or directory to be improved.

Predicted use cases:
    1. Code refactoring: The script can be used to refactor code by improving its syntax, formatting, and coverage.
    2. Code review: The script can be used to review code by checking for syntax errors, formatting issues, and coverage gaps.
    3. Code generation: The script can be used to generate improved code based on a given input.

Proposed features:
    1. Support for multiple programming languages
    2. Integration with code review tools
    3. Support for custom code improvement rules
    4. Integration with version control systems for automated improvements
    5. Machine learning models to predict potential bugs
    6. Suggestions for performance improvements
    7. Integration with continuous integration/continuous deployment (CI/CD) pipelines
    8. Improved user interface for non-technical users
    9. Real-time code improvement suggestions while coding
    10. Collaboration features for team-based code improvements
    11. Automated documentation generation
    12. Code improvement analytics dashboard
    13. Support for different coding styles and guidelines
    14. Enhanced security checks and vulnerability detection
    15. Code snippet recommendations from a centralized database
'''
