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

    def query(self, prompt, temperature=0.3, max_tokens=10000):
        response = requests.post(self.api_url, headers=self.headers, json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        })
        try:
            output = response.json()
            if output and isinstance(output, list) and 'generated_text' in output[0]:
                return output[0]['generated_text'].strip()
            return "Error: No valid response from API."
        except Exception as e:
            return f"API error: {str(e)}"

    # Agent prompt generators
    def run_quality_analysis(self, code, repo_name):
        prompt = f"""You are a code reviewer. Analyze this code from {repo_name} for general quality:
- Is the code readable and modular?
- Are naming conventions followed?
- Is it well-structured?

Code:
{code}
"""
        return self.query(prompt)

    def run_bug_detection(self, code):
        prompt = f"""You are a bug detection expert. Find and explain bugs or errors in the following code:
{code}
"""
        return self.query(prompt)

    def run_optimization(self, code):
        prompt = f"""You are a performance expert. Suggest memory or speed optimizations for this code:
{code}
"""
        return self.query(prompt)

    def run_report_generation(self, quality, bugs, optimizations, repo_name):
        prompt = f"""Generate a final markdown code review report for {repo_name} using the findings:

### Quality Analysis:
{quality}

### Bug Detection:
{bugs}

### Optimization Suggestions:
{optimizations}
"""
        return self.query(prompt, max_tokens=10000)
