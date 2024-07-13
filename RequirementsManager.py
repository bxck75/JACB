import os
import re
class RequirementsManager:
    def __init__(self, requirements_file, inscope_scripts=[]):
        self.requirements_file = requirements_file
        self.inscope_scripts = inscope_scripts
        self.requirements = self.get_requirements()
        self.requirements_list = self.get_requirements_list()

    def get_requirements(self):
        with open(self.requirements_file, 'r') as f:
            requirements = f.read()
        return requirements
    def get_requirements_list(self):
        requirements_list = re.findall(r'(\w+)', self.requirements)
        return requirements_list
    
    def get_imports(self, file_path):
        """Get a list of imported modules from a Python file."""
        imports = set()
        with open(file_path, 'r') as file:
            content = file.read()
            # Find all imports
            matches = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', content, re.MULTILINE)
            for match in matches:
                # Get the top-level module
                top_level_module = match.split('.')[0]
                imports.add(top_level_module)
        return imports

    def filter_requirements(self):
        """Filter the requirements.txt to include only used packages."""
        used_packages = set()
        for script_path in self.inscope_scripts:
            used_packages.update(self.get_imports(script_path))
        
        with open(self.requirements_path, 'r') as file:
            lines = file.readlines()
        
        filtered_requirements = []
        for line in lines:
            package = line.split('==')[0]
            if package in used_packages:
                filtered_requirements.append(line)
        
        with open('filtered_requirements.txt', 'w') as file:
            file.writelines(filtered_requirements)
        print("Filtered requirements.txt has been created as filtered_requirements.txt")

if __name__ == '__main__':
    # Define the path to your requirements.txt and script files
    requirements_path = 'requirements.txt'
    script_paths = ['/nr_ywo/coding/chatbot/FaissStorage.py',
                    '/nr_ywo/coding/chatbot/uber_toolkit_class.py'
                    ]  # Add paths to all your Python scripts
    app=RequirementsManager()
    app.filter_requirements(requirements_path, script_paths)