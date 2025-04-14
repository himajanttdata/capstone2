import os
import requests
from dotenv import load_dotenv

# Load API Key
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

GPT_AZURE_KEY = os.getenv("GPT_AZURE_KEY")
GPT_AZURE_ENDPOINT = os.getenv("GPT_AZURE_ENDPOINT")
GPT_AZURE_DEPLOYMENT = os.getenv("GPT_AZURE_DEPLOYMENT")
GPT_AZURE_API_VERSION = os.getenv("GPT_AZURE_API_VERSION", "2024-12-01-preview")

# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

class ResponderEngine:
    def __init__(self):
        self.api_url = (
            f"{GPT_AZURE_ENDPOINT}openai/deployments/"
            f"{GPT_AZURE_DEPLOYMENT}/chat/completions"
            f"?api-version={GPT_AZURE_API_VERSION}"
        )
        self.headers = {
            "Content-Type": "application/json",
            "api-key": GPT_AZURE_KEY,
        }
        # self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        # self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        # self.api_url = "http://localhost:11434/api/generate"
        # self.model = "llama3.2"
    

    def query(self, prompt: str, system_prompt: str):
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            # "max_tokens": 2000
        }
        try:
            response = requests.post(self.api_url,headers= self.headers, json= body)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Azure OpenAI: {str(e)}"


    # HF_API
    # def query(self, prompt, temperature=0.3, max_tokens=10000):
    #     response = requests.post(self.api_url, model = self.model_name, json={
    #         "inputs": prompt,
    #         "parameters": {
    #             "max_new_tokens": max_tokens,
    #             "temperature": temperature,
    #             "return_full_text": False
    #         }
    #     })
    #     try:
    #         output = response.json()
    #         if output and isinstance(output, list) and 'generated_text' in output[0]:
    #             return output[0]['generated_text'].strip()
    #         return "Error: No valid response from API."
    #     except Exception as e:
    #         return f"API error: {str(e)}"


    #OLLAMA MODEL
    # def query(self, prompt, max_tokens=1024):
    #     payload = {
    #         "model": self.model,
    #         "prompt": prompt,
    #         "stream": False,
    #         "options": {
    #             "temperature": 0.3,
    #             "num_predict": max_tokens
    #         }
    #     }

    #     try:
    #         response = requests.post(self.api_url, json=payload)
    #         response.raise_for_status()
    #         data = response.json()
    #         return data.get("response", "Error: No response field found.")
    #     except Exception as e:
    #         return f"Error communicating with Ollama: {str(e)}"




    # Agent prompt generators
    def run_quality_analysis(self, code, repo_name):
        system_prompt = f"""You are a code reviewer. Analyze this code from {repo_name} for general quality and provide feedback on:
- Is the code readable and modular?
- Are naming conventions followed?
- Is it well-structured?
provide feedback in simple way. don't get in so deep
"""
        prompt = f""" Code to Analyze: {code}"""
        return self.query(prompt,system_prompt)

    def run_bug_detection(self, code):
        system_prompt = """You are a bug detection expert. Analyze the following code and identify any possible bugs, logic errors, or syntax issues.
        For each bug mention in which line of the code the bug is present, explain the issue and suggest a fix."""
        prompt = f"""code:{code}"""
        return self.query(prompt, system_prompt)

    def run_optimization(self, code):
        system_prompt = """You are a performance Optimization expert. Suggest optimizations to improve speed, memory usage, or best practices for the following code.
        Only include practical improvements. Also give the optimized version of the code."""
        prompt = f"""code:{code}"""
        return self.query(prompt,system_prompt)

    def run_report_generation(self, quality, bugs, optimizations, repo_name):
        system_prompt = f"""
        you are the report generator. Using the analysis results below, create a detailed and structured markdown review report for the repository "{repo_name}"""
        prompt = f"""
### Quality Analysis:
{quality}

### Bug Detection:
{bugs}

### Optimization Suggestions:
{optimizations}

###Summary:

- Code Quality: [Excellent / Good / Needs Improvement]
- Bugs: [None/Minor/Major]
- Optimization: [Not Required/ Recommended/ Essential]
###Conclusion:
Provide a conclusion on whether the code is Production-ready or needs further work.
Follow this template given STRICTLY do not add any other sections.
"""
        return self.query(prompt,system_prompt)
