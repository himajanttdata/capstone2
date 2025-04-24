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

    def query(self, prompt: str, system_prompt: str):
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=body)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Azure OpenAI: {str(e)}"

    def run_quality_analysis(self, code, repo_name):
        system_prompt = f"""You are a code reviewer. Analyze this code from {repo_name} and provide a brief summary of its quality in terms of structure, readability, and naming.
        Use no more than 4 bullet points. No improvement suggestions."""
        prompt = f"""Code to Analyze: {code}"""
        return self.query(prompt, system_prompt)

    def run_bug_detection(self, code):
        system_prompt = """You are a bug detection expert. Analyze the following code and identify any possible bugs, logic errors, or syntax issues.

                        For each issue:
                        1. **Specify the line number** where the bug occurs.
                        2. **Provide the exact code** causing the issue on that line.
                        3. **Explain the problem** clearly.
                        4. Suggest a **fixed version of the code** for that specific line.

                        Ensure that you are providing clear, detailed explanations of why the code is problematic, and offer the correct way to fix it."""
        prompt = f"code: {code}"
        return self.query(prompt, system_prompt)

    def run_optimization(self, code):
        system_prompt = """You are an optimization expert. Summarize 3–5 practical suggestions to improve performance, memory usage, or fix potential security issues.
        Keep each point short and output in bullet-point HTML format."""
        prompt = f"code: {code}"
        return self.query(prompt, system_prompt)

#     def run_standards_compliance(self, code):
#         system_prompt = """You are a Standard Compliance Agent. Your task is to review the provided code and ensure it adheres to the coding standards and best practices of the respective programming language.
#          The languages and their standards are as follows:
#          Python: PEP8
#         Java: Google Java Style Guide
#         JavaScript: Airbnb JavaScript Style Guide
#         C++: C++ Core Guidelines
#         HTML: W3C HTML5 Specification"""
#         prompt = f"code: {code}"
#         return self.query(prompt, system_prompt)

#     def run_security_analysis(self, code):
#         system_prompt = """You are a Security Analysis Agent. Your task is to review the provided code and ensure it is secure,
#           free from data leaks, memory leaks, and vulnerabilities. The languages and their security standards are as follows:
#           Python: OWASP Python Security Guidelines
#         Java: OWASP Java Security Guidelines
#         JavaScript: OWASP JavaScript Security Guidelines
#         C++: CERT C++ Secure Coding Standard
#         HTML: OWASP HTML Security Guidelines"""
#         prompt = f"code: {code}"
#         return self.query(prompt, system_prompt)

#     def run_docstring_generation(self, code):
#         system_prompt = """You are a code documentation assistant. Add or improve docstrings for all functions and classes in the given code.
# Ensure clarity and completeness."""
#         prompt = f"code: {code}"
#         return self.query(prompt, system_prompt)

    def run_unit_test_suggestions(self, code):
        system_prompt = """You are a unit test case suggestion agent. Suggest up to 5 essential unit test cases for the given code using the respective testing framework.
        Keep suggestions short. Output should be in HTML list format."""
        prompt = f"code: {code}"
        return self.query(prompt, system_prompt)
    
    def run_final_code_generator(self, code, quality, bugs, optimizations):
        system_prompt =  f"""You are a senior software engineer. Create a concise and optimized version of the original code, based on the provided quality issues, bug fixes, and optimizations.
        Limit comments and use clean formatting. Return code as an HTML-formatted code block. If the {code} is HTML then generate the optimized code for this section and return it in an ESCAPED FORMAT,
        so it can be rendered as visible code inside a <pre><code> block in HTML."""
        prompt = f"""
        code: {code}

        --- Reviews ---
        Quality: {quality}
        Bugs: {bugs}
        optimizations: {optimizations}
        return self.query(prompt, system_prompt)"""

        return self.query(prompt, system_prompt)

    def run_report_generation(self, quality, bugs, optimizations, tests, final_code, repo_name):
        system_prompt = f"""You are an HTML report generator. Generate a clean, minimal HTML report for the repo {repo_name} using the following sections: Quality Analysis, Bug Detection, Optimizations, Test Suggestions, Final Code, Summary, and Conclusion.

                        Strict HTML Formatting Rules:
                        - Use <h3> for main sections and <h4> for sub-sections.
                        - Use <ul><li> for bullet points.
                        - Highlight keywords with <strong>.
                        - Format code with <pre><code class='language-{{lang}}'></code></pre>.
                        - DO NOT use Markdown or HTML table tags.
                        - DO NOT invent or skip any section.

                        Ensure the report is short and structured for easy rendering on a UI.
                        """
        prompt = f"""
                        <h3>Quality Analysis</h3>
                        {quality}

                        <h3>Bug Detection</h3>
                        {bugs}

                        <h3>Optimization Suggestions</h3>
                        {optimizations}

                        <h3>Unit Test Suggestions</h3>
                        {tests}

                        <h3>Final Optimized Code</h3>
                        <pre><code class='language-python'>
                        {final_code}
                        </code></pre>

                        <h3>Summary</h3>
                        <ul>
                        <li><strong>Code Quality:</strong> [Excellent / Good / Needs Improvement]</li>
                        <li><strong>Bugs:</strong> [None / Minor / Major]</li>
                        <li><strong>Optimizations:</strong> [None / Minor / Major]</li>
                        <li><strong>Testing:</strong> [Well Covered / Needs More Tests / No Tests]</li>
                        </ul>

                        <h3>Conclusion</h3>
                        <p>Provide a short recommendation on whether the code is production-ready, needs revisions, or should be refactored.</p>
                        """

        
        return self.query(prompt, system_prompt)
