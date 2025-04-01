from flask import Flask, render_template, request, jsonify, send_file
from github_engine import GitHubEngine
from responder_engine import ResponderEngine
from docx import Document  # Import python-docx for DOCX generation
import os

app = Flask(__name__)
UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    review_report = None
    if request.method == "POST":
        if "repo_url" in request.form and request.form["repo_url"]:
            # Process GitHub repository
            repo_url = request.form["repo_url"]
            github_engine = GitHubEngine()
            repo_name, all_chunks = github_engine.process_repository(repo_url)

        elif "file" in request.files:
            # Process uploaded file
            uploaded_file = request.files["file"]
            if uploaded_file.filename:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
                uploaded_file.save(file_path)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()

                from template_reader import TemplateReader
                template_reader = TemplateReader()
                all_chunks = template_reader.chunk_code(file_content)
                repo_name = uploaded_file.filename  # Use filename as repo_name

        if all_chunks:
            responder = ResponderEngine()
            review_report = responder.generate_code_review(repo_name, all_chunks)

    return render_template("index.html", review=review_report)


@app.route("/download_docx", methods=["POST"])
def download_docx():
    data = request.get_json()
    review_text = data.get("review", "No review available.")

    doc = Document()
    doc.add_heading("Code Review Report", level=1)
    doc.add_paragraph(review_text)

    docx_path = "review_report.docx"
    doc.save(docx_path)

    return send_file(docx_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
