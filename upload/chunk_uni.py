import os
import shutil
import tempfile
import git
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pathlib import Path

def clone_github_repo(repo_url, dest_dir):
    """Clones a GitHub repository to the specified destination directory."""
    try:
        git.Repo.clone_from(repo_url, dest_dir)
        print("Repository cloned successfully!")
        return dest_dir
    except Exception as e:
        print("Error cloning repository:", e)
        return None

def list_code_files(directory):
    """Lists all code-related files in the given directory."""
    extensions = {'.py', '.java', '.js', '.html', '.css', '.cpp', '.c', '.ts'}
    code_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix in extensions:
                code_files.append(os.path.join(root, file))
    return code_files

def read_file_content(file_path):
    """Reads the content of a given file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def universal_chunking(file_path, chunk_size=100):
    """Chunks the code file based on a universal approach."""
    content = read_file_content(file_path)
    if not content:
        return []
    
    # Tokenize by lines and split into chunks
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        current_chunk.append(line)
        current_length += len(line)
        if current_length >= chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
    
    return chunks

def embed_chunks(chunks, model_name='BAAI/bge-base-en-v1.5'):
    """Generates embeddings for code chunks using a transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

def store_in_faiss(embeddings):
    """Stores embeddings in a FAISS index."""
    d = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, "code_embeddings.index")
    print("Embeddings stored in FAISS index successfully!")
    return index

def process_repository(repo_url):
    """Main function to clone, chunk, embed, and store code from a GitHub repo."""
    temp_dir = tempfile.mkdtemp()
    repo_path = clone_github_repo(repo_url, temp_dir)
    if not repo_path:
        return None
    
    code_files = list_code_files(repo_path)
    all_chunks = []
    for file in code_files:
        all_chunks.extend(universal_chunking(file))
    
    embeddings = embed_chunks(all_chunks)
    store_in_faiss(embeddings)
    # shutil.rmtree(temp_dir)  # Cleanup after processing
    print("Processing completed successfully!")
    
    return "Processing completed!"

def process_uploaded_files(uploaded_dir):
    """Main function to process directly uploaded files or folders."""
    code_files = list_code_files(uploaded_dir)
    all_chunks = []
    for file in code_files:
        all_chunks.extend(universal_chunking(file))
    
    embeddings = embed_chunks(all_chunks)
    store_in_faiss(embeddings)
    print("Uploaded files processed successfully!")
    
    return "Processing completed!"

def main():
    """Main entry point for the script."""
    choice = input("Enter 1 to process a GitHub repo, 2 to process uploaded files: ")
    if choice == "1":
        repo_url = input("Enter the GitHub repository URL: ")
        process_repository(repo_url)
    elif choice == "2":
        uploaded_dir = input("Enter the path to the uploaded files or folder: ")
        process_uploaded_files(uploaded_dir)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
