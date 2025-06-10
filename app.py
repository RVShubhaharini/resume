import os
import re
import zipfile
import tempfile
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
import docx

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langgraph.graph import StateGraph, START, END

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MODEL_ID = "google/flan-t5-Small"  # Using seq2seq model

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLM pipeline
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

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
'''
def feedback_agent(state):
    resume_text = state["resume_text"]
    job_desc = state["job_desc"]
    prompt = f"""Given the following job description:
{job_desc}

Provide clear, helpful bullet-point feedback to improve this resume:
{resume_text}
"""
    try:
        response = llm_pipe(prompt)[0]['generated_text'].strip()
    except Exception as e:
        response = f"Unable to generate feedback due to error: {str(e)}"
    state["feedback"] = response
    return state
'''
def feedback_agent(state):
    resume_text = state.get("resume_text", "")[:3000]  # prevent overload
    job_desc = state.get("job_desc", "")[:1000]

    prompt = (
        "You are an expert resume reviewer.\n\n"
        "Given this job description:\n"
        f"{job_desc}\n\n"
        "And this resume:\n"
        f"{resume_text}\n\n"
        "Provide clear, professional feedback in bullet points to help the candidate improve their resume."
    )

    try:
        output = llm_pipe(prompt)[0]["generated_text"]
        state["feedback"] = output.strip()
    except Exception as e:
        state["feedback"] = f"Error generating feedback: {str(e)}"
    
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

@app.route("/shortlist", methods=["POST"])
def shortlist():
    try:
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

                state = {
                    "resume_text": resume_text,
                    "job_desc": job_desc
                }
                result = graph.invoke(state)
                all_scores.append({
                    "filename": filename,
                    "text": resume_text,
                    "metadata": result.get("features", {}),
                    "score": result.get("score", 0),
                    "hr_eval": result.get("hr_eval", {}),
                    "feedback": result.get("feedback", "")
                })

        top3 = sorted(all_scores, key=lambda x: x['score'], reverse=True)[:3]
        filenames = [r["filename"] for r in top3]

        return jsonify({
            "resumes": top3,
            "filenames": filenames
        })
    except Exception as e:
        print("Error during resume processing:", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/download", methods=["POST"])
def download_zip():
    data = request.get_json()
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided"}), 400

    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in filenames:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=fname)

    return send_file(zip_path, as_attachment=True, download_name="shortlisted_resumes.zip")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route("/")
def index():
    return render_template("index.html")  # Optional: if you have a frontend

if __name__ == "__main__":
    app.run(debug=True, port=5000)
