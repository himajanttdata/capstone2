import os
import git
import glob
import faiss
import numpy as np
from template_reader import TemplateReader
from doc_generation import DocGeneration

class GitHubEngine:
    def __init__(self, base_dir="repos"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def get_repo_name(self, repo_url):
        return repo_url.split("/")[-1].replace(".git", "")

    def clone_repo(self, repo_url):
        repo_name = self.get_repo_name(repo_url)
        repo_path = os.path.join(self.base_dir, repo_name)

        if os.path.exists(repo_path):
            print(f"Repo '{repo_name}' already cloned.")
            return repo_name, repo_path

        print(f"Cloning repo: {repo_url} ...")
        git.Repo.clone_from(repo_url, repo_path)
        return repo_name, repo_path

    def get_code_files(self, repo_path):
        extensions = [".py", ".java", ".js", ".cpp", ".html"]
        code_files = []
        for ext in extensions:
            code_files.extend(glob.glob(f"{repo_path}/**/*{ext}", recursive=True))
        return code_files

    def process_repository(self, repo_url):
        repo_name, repo_path = self.clone_repo(repo_url)
        code_files = self.get_code_files(repo_path)

        template_reader = TemplateReader()
        all_chunks = []
        for file in code_files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
                chunks = template_reader.chunk_code(code)
                all_chunks.extend(chunks)

        doc_gen = DocGeneration()
        doc_gen.store_vectors(repo_name, all_chunks)

        return repo_name, all_chunks
