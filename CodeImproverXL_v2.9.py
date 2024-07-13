
import sys
import os
import re
import autopep8
import coverage
import random
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from LLMChatBot import LLMChatBot
from FaissStorage import AdvancedVectorStore
sys.path.append(str(Path(__file__).parent.parent.parent.parent)) # protected
from system_prompts import __all__ as prompts                   # protected
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
        self.python_mark: Optional[str] = r"```(python|py|)\n(?P<code>[\s\S]+?)\n```"
        self.llm = LLMChatBot(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.avs = AdvancedVectorStore(email=os.getenv("EMAIL"), password=os.getenv("PASSWD"))
        self.current_conversation
        self.avs.logger.info("Init!")

    @staticmethod
    def read_code(text: str, code_mark: str) -> Optional[str]:
        match = re.search(code_mark, text.content)
        if match:
            return match.group("code")
        return None

    def save_changes(self, text: str) -> None:
        with open(self.log_file, "a") as file:
            file.write(text)
        self.avs.logger.info(f"Changes saved to log {self.log_file}")

    def check_syntax_errors(self, code: str) -> None:
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            self.avs.logger.info(f"Syntax Error: {e}")

    def format_code(self, code: str) -> str:
        return autopep8.fix_code(code)

    def generate_coverage_report(self, code: str) -> None:
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

    def improve_code(self, path: Optional[str] = None) -> None:
        """Improves the code in a given file or all files in a given directory."""
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = input("[Enter the path of the file you want to improve (enter to self-improve):]")

        if path:
            if os.path.isfile(path):
                paths = [path]
            elif os.path.isdir(path):
                paths = list(Path(path).rglob("*.py"))
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
                                       ```py
                                       {code}
                                       ```"""
                # Result of the llm predicting improvements
                results = str(self.llm(user_task_prompt))
                
                if isinstance(results, list):
                    results = str(''.join([str(msg) for msg in results]))
                
                code = self.read_code(results, self.python_mark)
                
                if code:
                    self.check_syntax_errors(code)
                    code = self.format_code(code)
                    self.generate_coverage_report(code)
                    
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
""" 
I made the following changes:
    1. Added type hints for function parameters and return types.
    2. Refactored the code to make it more readable and maintainable.
    3. Improved the logging and error handling.
    4. Added a description, usage, and predicted use cases to the comment block.
    5. Proposed new features for future development. """