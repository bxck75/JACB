import sys
import subprocess
import requests
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from bs4 import BeautifulSoup

'''
TODOS:
    - make a extra frame in the page where onclick of a package the dir() of it shows
    - make a tab for requirements.txt files can be made/edited/managed
'''

class PackageManager:
    def __init__(self, temp_requirements: str = 'temp_requirements.txt'):
        self.temp_requirements = temp_requirements
    
    def search_packages(self, query: str) -> list:
        url = f"https://pypi.org/search/?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            packages = self.extract_package_names(response.text)
            return packages
        else:
            messagebox.showerror("Error", "Failed to search packages on PyPI")
            return []

    def extract_package_names(self, html_content: str) -> list:
        soup = BeautifulSoup(html_content, 'html.parser')
        package_tags = soup.find_all('a', class_='package-snippet')
        packages = [tag.find('span', class_='package-snippet__name').text for tag in package_tags]
        return packages

    def install_package(self, package_name: str) -> None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            with open(self.temp_requirements, 'a') as f:
                f.write(f"{package_name}\n")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to install package: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def uninstall_package(self, package_name: str) -> None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package_name])
            with open(self.temp_requirements, 'r+') as f:
                lines = f.readlines()
                f.seek(0)
                for line in lines:
                    if package_name not in line:
                        f.write(line)
                f.truncate()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to uninstall package: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_package_info(self, package_name: str) -> str:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        if response.status_code == 200:
            info = response.json()["info"]
            return f"Version: {info['version']}\nDescription: {info['description']}"
        else:
            messagebox.showerror("Error", f"Failed to get package info for {package_name}")
            return ""

class PackageGUI:

    version: float = 0.2

    def __init__(self, root: tk.Tk, manager: PackageManager):
        self.manager = manager
        self.search_var = tk.StringVar()
        self.setup_ui(root)

    def setup_ui(self, root: tk.Tk) -> None:
        root.title("Python Package Manager")
        
        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        search_label = tk.Label(frame, text="Search for a package:")
        search_label.pack(anchor=tk.W)

        search_entry = tk.Entry(frame, textvariable=self.search_var, width=50)
        search_entry.pack(anchor=tk.W)
        
        search_button = tk.Button(frame, text="Search", command=self.search_packages)
        search_button.pack(anchor=tk.W, pady=5)
        
        self.package_list = tk.Listbox(frame, selectmode=tk.SINGLE, width=50, height=15)
        self.package_list.pack(anchor=tk.W)
        
        install_button = tk.Button(frame, text="Install Selected Package", command=self.install_package)
        install_button.pack(anchor=tk.W, pady=5)
        
        uninstall_button = tk.Button(frame, text="Uninstall Selected Package", command=self.uninstall_package)
        uninstall_button.pack(anchor=tk.W, pady=5)
        
        info_button = tk.Button(frame, text="Get Package Info", command=self.get_package_info)
        info_button.pack(anchor=tk.W, pady=5)

    def search_packages(self) -> None:
        query = self.search_var.get()
        if query:
            packages = self.manager.search_packages(query)
            self.package_list.delete(0, tk.END)
            for package in packages:
                self.package_list.insert(tk.END, package)
    
    def install_package(self) -> None:
        selected_index = self.package_list.curselection()
        if selected_index:
            package_name = self.package_list.get(selected_index)
            self.manager.install_package(package_name)
            messagebox.showinfo("Success", f"Package '{package_name}' installed successfully")

    def uninstall_package(self) -> None:
        selected_index = self.package_list.curselection()
        if selected_index:
            package_name = self.package_list.get(selected_index)
            self.manager.uninstall_package(package_name)
            messagebox.showinfo("Success", f"Package '{package_name}' uninstalled successfully")

    def get_package_info(self) -> None:
        selected_index = self.package_list.curselection()
        if selected_index:
            package_name = self.package_list.get(selected_index)
            info = self.manager.get_package_info(package_name)
            messagebox.showinfo("Package Info", info)




if __name__ == "__main__":
    root = tk.Tk()
    manager = PackageManager()
    app = PackageGUI(root, manager)
    root.mainloop()

'''
Description:
    A Python package manager with a GUI that allows users to search for packages on PyPI, install them, uninstall them, and display package information.

Usage:
    Run the script, and a GUI window will appear. Enter a package name in the search field and click the "Search" button. 
    A list of matching packages will appear in the list box. Select a package and click the "Install Selected Package" button to install it, 
    click the "Uninstall Selected Package" button to uninstall it, or click the "Get Package Info" button to display package information.

Predicted use cases:
    - Personal use for managing Python packages
    - In a development team for easy package management
    - In a CI/CD pipeline for automating package installation

Proposed features:

'''