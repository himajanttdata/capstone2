import os
import requests
from dotenv import load_dotenv

# Load API Key
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

class ResponderEngine:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    def query(self, payload):
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        return response.json()

    def generate_code_review(self, repo_name, reconstructed_code):
        if not reconstructed_code:
            return f"No code found for analysis in {repo_name}."

        print(f"Generating review report for repository: {repo_name}")

        system_prompt = f"""
        You are an expert code reviewer. Analyze the provided code and provide a structured review.

        **Repository: {repo_name}**

        **Review Guidelines:**
        - Identify **only real issues** (e.g., syntax errors, security vulnerabilities, accessibility concerns).
        - **Highlight maintainability and readability** improvements.
        - Provide **clear and concise** suggestions.
        - **Avoid unnecessary changes** (e.g., changing working code for no reason).

        **Review Structure:**
        1. **Correctness Assessment**: Highlight syntax errors or logical flaws.
        2. **Optimization Suggestions**: Performance, memory usage, and efficiency tips.
        3. **Best Practices Check**: Code clarity, modularity, naming conventions.
        4. **Security Analysis**: Identify vulnerabilities (e.g., injections, bad access controls).
        5. **Optimized Version (if needed)**: Provide corrections **only if essential**.
        """

        full_prompt = f"{system_prompt}\n\nCode to review:\n{reconstructed_code}"

        output = self.query({
            "inputs": full_prompt,
            "parameters": {"max_new_tokens": 1024, "return_full_text": False, "temperature": 0.3}
        })

        if output and isinstance(output, list) and 'generated_text' in output[0]:
            return output[0]['generated_text'].strip()

        return "Error: No valid response from API."
