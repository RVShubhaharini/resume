import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langgraph.graph import StateGraph, START, END

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

app = Flask(
    __name__,
    static_folder='resume-frontend/build',
    static_url_path=''
)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLM pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_resume_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    else:
        return ""

# --- AGENT NODE FUNCTIONS ---
def recruiter_agent(state):
    resume_text = state["resume_text"]
    education = re.findall(r"(B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|MBA|PhD|Bachelor|Master|Doctor)", resume_text, re.I)
    skills = re.findall(r"(Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js)", resume_text, re.I)
    experience = re.findall(r"(\d+)\s+years?", resume_text)
    state["features"] = {
        "education": list(set(education)),
        "skills": list(set(skills)),
        "experience": max([int(e) for e in experience], default=0)
    }
    return state

def analyst_agent(state):
    features = state["features"]
    job_desc = state["job_desc"]
    jd_skills = re.findall(r"(Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js)", job_desc, re.I)
    skill_overlap = len(set(features['skills']).intersection(set(jd_skills)))
    experience_score = min(features['experience'], 10)
    state["score"] = skill_overlap * 10 + experience_score * 5
    return state

def hr_agent(state):
    resume_text = state["resume_text"]
    soft_skills = re.findall(r"(leadership|communication|teamwork|initiative|adaptability|problem-solving)", resume_text, re.I)
    red_flags = re.findall(r"(terminated|fired|gap|unemployed)", resume_text, re.I)
    state["hr_eval"] = {
        "soft_skills": list(set(soft_skills)),
        "red_flags": list(set(red_flags))
    }
    return state

def feedback_agent(state):
    resume_text = state["resume_text"]
    job_desc = state["job_desc"]
    prompt = f"Given the following job description:\n{job_desc}\n\nProvide detailed, actionable feedback to improve this resume:\n{resume_text}\nFeedback:"
    result = llm_pipe(prompt)[0]['generated_text']
    state["feedback"] = result
    return state

# --- BUILD LANGGRAPH ---
def build_graph():
    builder = StateGraph(dict)
    builder.add_node("recruiter_agent", recruiter_agent)
    builder.add_node("analyst_agent", analyst_agent)
    builder.add_node("hr_agent", hr_agent)
    builder.add_node("feedback_agent", feedback_agent)
    builder.add_edge(START, "recruiter_agent")
    builder.add_edge("recruiter_agent", "analyst_agent")
    builder.add_edge("analyst_agent", "hr_agent")
    builder.add_edge("hr_agent", "feedback_agent")
    builder.add_edge("feedback_agent", END)
    return builder.compile()

graph = build_graph()

# --- FLASK ROUTES ---
@app.route("/shortlist", methods=["POST"])
def shortlist():
    job_desc = request.form.get("job_description", "")
    files = request.files.getlist("resumes")
    if not job_desc or not files:
        return jsonify({"error": "Missing job description or files"}), 400

    all_scores = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            resume_text = extract_resume_text(save_path)

            # Run through LangGraph
            state = {
                "resume_text": resume_text,
                "job_desc": job_desc
            }
            result = graph.invoke(state)
            all_scores.append({
                "filename": filename,
                "metadata": result.get("features", {}),
                "score": result.get("score", 0),
                "hr_eval": result.get("hr_eval", {}),
                "feedback": result.get("feedback", ""),
            })

    # Recommend top 3
    top3 = sorted(all_scores, key=lambda x: x['score'], reverse=True)[:3]
    for entry in top3:
        entry["download_url"] = f"/uploads/{entry['filename']}"
    return jsonify(top3)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)

