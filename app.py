from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from github_engine import GitHubEngine
from responder_engine import ResponderEngine
from docx import Document  # Import python-docx for DOCX generation
from agent_graph import app as review_graph
import os
import uuid
import re

app = Flask(__name__)
UPLOAD_FOLDER = "upload"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER
# @app.route("/", methods=["GET", "POST"])
# def index():
#     review_report = None
#     if request.method == "POST":
#         if "repo_url" in request.form and request.form["repo_url"]:
#             # Process GitHub repository
#             repo_url = request.form["repo_url"]
#             github_engine = GitHubEngine()
#             repo_name, all_chunks = github_engine.process_repository(repo_url)

#         elif "file" in request.files:
#             # Process uploaded file
#             uploaded_file = request.files["file"]
#             if uploaded_file.filename:
#                 file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
#                 uploaded_file.save(file_path)

#                 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                     file_content = f.read()

#                 from template_reader import TemplateReader
#                 template_reader = TemplateReader()
#                 all_chunks = template_reader.chunk_code(file_content)
#                 repo_name = uploaded_file.filename  # Use filename as repo_name

#         if all_chunks:
#             responder = ResponderEngine()
#             review_report = responder.generate_code_review(repo_name, all_chunks)

#     return render_template("intro1.html", review=review_report)
@app.route("/")
def intro():
    return render_template("intro2.html")
@app.route("/index2")
def index1():
    return render_template("index2.html")
@app.route("/review2")
def review1():
    # Load review from saved file
    report_path = os.path.join(REPORT_FOLDER,"latest_review.txt")
    if not os.path.exists(report_path):
        return render_template("review2.html", review = None)
    with open(report_path, "r", encoding="utf-8") as f:
        review_text = f.read()
    
    sections = re.split(r"^##\s+\**(.*?)\**\s*$", review_text, flags = re.MULTILINE)

    section_map = {}
    for i in range(1, len(sections), 2):
        title = sections[i].strip().lower()
        body = sections[i+1].strip()
        section_map[title.lower().replace(" ","_")] = {
            "title": title,
            "body": body
        }
    # print("sections content: ",sections)
    return render_template("review2.html", sections = section_map)
    # review = {
    #     "quality" : "",
    #     "bugs" : "",
    #     "suggestions" : "",
    #     "optimized_code" : "",
    #     "summary" : "",
    #     "conclusion": ""
    # }

    # sections = [
    #     "Quality Analysis:",
    #     "Bug Detection:",
    #     "Optimization Suggestions:",
    #     "Optimized code:",
    #     "Summary:",
    #     "Conclusion:"
    # ]

    # # lines = review_text.splitlines()
    # current_section = None

    # section_map = {
    #     "Quality Analysis:": "quality",
    #     "Bug Detection:": "bugs",
    #     "Optimization Suggestions": "suggestions",
    #     "Optimized code:": "optimized_code",
    #     "Summary:": "summary",
    #     "Conclusion": "conclusion"

    # }
    # # sections = {key: "" for key in section_map.values()}
    # # current_section = None
    # for line in review_text.splitlines():
    #     # match = re.match(r"\*\*(.*?)\*\*", line.strip())
    #     # if match:
    #     #     heading = match.group(1).lower()
    #     #     for key in section_map:
    #     #         if key in heading:
    #     #             current_section = section_map[key]
    #     #             break
    #     # elif current_section:
    #     #     sections[current_section] += line + "\n"

    #     line = line.strip()
    #     if line.startswith("**") and line.endswith("**:"):
    #         section_title = line.strip("* ")
    #         current_section = section_map.get(section_title + ":")
    #     elif line.startswith("**") and line.endswith(":**"):
    #         section_title = line.strip("* ")
    #         current_section = section_map.get(section_title)
    #     elif current_section:
    #         review[current_section] += line + "\n"
    # # print("Optimized version : ")
    # # print(review["code"])
    # # print(sections["Optimized Version"])     
    # return render_template("review2.html",review = review)

# Handle Submission (POST from index1)
@app.route("/submit", methods=["POST"])
def submit():
    # all_chunks = None
    # repo_name = "Uploaded_Code"
    review_report = None

    if "repo_url" in request.form and request.form["repo_url"]:
        repo_url = request.form["repo_url"]
        github_engine = GitHubEngine()
        repo_name, all_chunks = github_engine.process_repository(repo_url)

    elif "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()

            from template_reader import TemplateReader
            template_reader = TemplateReader()
            all_chunks = template_reader.chunk_code(file_content)
            repo_name = uploaded_file.filename

    if all_chunks:
        # responder = ResponderEngine()
        result = review_graph.invoke({
            "repo": repo_name,
            "code": all_chunks
        })

        review_report = result["final_report"]

        # Save the review text
        report_path = os.path.join(REPORT_FOLDER, "latest_review.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(review_report)
    return jsonify({"success": True}) if review_report else jsonify({"error": "Failed to generate review"}),200



# @app.route("/download_docx", methods=["POST"])
# def download_docx():
#     data = request.get_json()
#     review_text = data.get("review", "No review available.")

#     doc = Document()
#     doc.add_heading("Code Review Report", level=1)
#     doc.add_paragraph(review_text)

#     docx_path = "review_report.docx"
#     doc.save(docx_path)

#     return send_file(docx_path, as_attachment=True)
# Download Review as DOCX
@app.route("/download_docx", methods=["GET"])
def download_docx():
    report_path = os.path.join(REPORT_FOLDER, "latest_review.txt")
    if not os.path.exists(report_path):
        return "Report not found", 404

    with open(report_path, "r", encoding="utf-8") as f:
        review_text = f.read()

    doc = Document()
    doc.add_heading("Code Review Report", level=1)
    doc.add_paragraph(review_text)

    docx_path = os.path.join(REPORT_FOLDER, "latest_review.docx")
    doc.save(docx_path)

    return send_file(docx_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
