import sys
import os
import re
import autopep8
import coverage
from pathlib import Path
from rich import print as rp
from rich.console import Console
from typing import Optional, Dict
from LLMChatBot import LLMChatBot
from AdvancedVectorStore import AdvancedVectorStore, LLMChatBot
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from system_prompts import __all__ as prompts
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class CodeImprover:

    version: Optional[float] = 2.9
    script_path: Optional[str] = None
    log_file: Optional[str] = os.path.join(f"{str(Path(__file__).parent)}", "changes.txt")

    def __init__(self):
        # ... (initialization code remains the same)

    def read_code(self, text: str, language: str) -> Optional[str]:
        # ... (method remains the same)

    def save_changes(self, text: str) -> None:
        # ... (method remains the same)

    def check_syntax_errors(self, code: str, language: str) -> None:
        # ... (method remains the same)

    def format_code(self, code: str, language: str) -> str:
        # ... (method remains the same)

    def generate_coverage_report(self, code: str, language: str) -> None:
        if language == "python":
            temp_file = f"{str(Path(__file__).parent)}/temp.py"
            try:
                with open(temp_file, "w") as file:
                    file.write(code)
                
                cov = coverage.Coverage(data_file=None)
                cov.start()
                
                # Execute the code in a separate process
                exit_code = os.system(f"python {temp_file}")
                if exit_code != 0:
                    self.avs.logger.error(f"Error executing the code. Exit code: {exit_code}")
                    return

                cov.stop()
                cov.save()

                # Generate report
                report = io.StringIO()
                cov.report(file=report, show_missing=True, skip_covered=True, ignore_errors=True)
                self.avs.logger.info(f"Coverage report:\n{report.getvalue()}")

            except Exception as e:
                self.avs.logger.error(f"Error generating coverage report: {str(e)}")
            finally:
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if os.path.exists(".coverage"):
                    os.remove(".coverage")

    def improve_code(self, path: Optional[str] = None, language: str = "python") -> None:
        if self.script_path and os.path.exists(self.script_path):
            path = self.script_path
        else:
            path = '/nr_ywo/coding/JACB/test_input/'

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
                
                results = self.llm(user_task_prompt)

                with open('temp_res.txt', 'w') as f:
                    rp(results, file=f)

                if isinstance(results, list):
                    results = ''.join([str(msg) for msg in results])
                
                code = self.read_code(str(results), language)
                
                if code:
                    self.check_syntax_errors(code, language)
                    code = self.format_code(code, language)
                    self.generate_coverage_report(code, language)
                    
                    new_file_path = str(Path(path).with_name(f"{Path(path).stem}_generated_{str(self.version).replace('.', '_')}_improvement{Path(path).suffix}"))
                    
                    with open(new_file_path, "w") as file:
                        file.write(code)
                    
                    changes = results.split("I made the following changes:")
                    if len(changes) > 1:
                        changes = f"I made the following changes in version {self.version}:\n{changes[1]}"
                        self.save_changes(changes)
                    else:
                        self.avs.logger.info("No changes detected.")
                
            except Exception as e:
                self.avs.logger.error(f"Error processing file {path}: {str(e)}")

if __name__ == "__main__":
    replicator = CodeImprover()
    replicator.improve_code(path="CodeImproverXL.py", language="python")
