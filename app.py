import os
import re
import zipfile
import io
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import PyPDF2
import docx
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# FLAN-T5 initialization (loads once)
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flan_model = flan_model.to(device)


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


embedder = SentenceTransformer("all-MiniLM-L6-v2")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_resume_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    return extract_text_from_pdf(file_path) if ext == "pdf" else extract_text_from_docx(file_path)

def list_cosine_sim(list1, list2):
    list1_text = " ".join([s.lower() for s in list1])
    list2_text = " ".join([s.lower() for s in list2])
    vec = CountVectorizer().fit_transform([list1_text, list2_text]).toarray()
    return float(round(cosine_similarity([vec[0]], [vec[1]])[0][0] * 100, 2))

# ---- AGENTS ----
def recruiter_agent(state):
    resume_text = state["resume_text"]
    education = re.findall(r"(B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|MBA|PhD|Bachelor|Master|Doctor)", resume_text, re.I)
    skills = re.findall(r"(Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js)", resume_text, re.I)
    experience_years = re.findall(r"(\d+)\s+years?", resume_text)
    
    state["features"] = {
        "education": list(set(education)),
        "skills": list(set(skills)),
        "experience": max([int(y) for y in experience_years], default=0)
    }
    return state

def analyst_agent(state):
    job_desc = state["job_desc"]
    resume_text = state["resume_text"]
    job_vec = embedder.encode([job_desc])
    resume_vec = embedder.encode([resume_text])
    sim = cosine_similarity(job_vec, resume_vec)[0][0]
    state["analyst_score"] = float(round(sim * 100, 2))
    return state

def skills_agent(state):
    skills_in_resume = state["features"]["skills"]
    job_skills = re.findall(r"(Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js)", state["job_desc"], re.I)
    score = list_cosine_sim(job_skills, skills_in_resume)
    state["skills_score"] = float(score)
    return state

def education_agent(state):
    resume_edu = state["features"]["education"]
    job_edu = re.findall(r"(B\.?Tech|M\.?Tech|MBA|PhD|Bachelor|Master)", state["job_desc"], re.I)
    score = list_cosine_sim(job_edu, resume_edu)
    state["education_score"] = float(score)
    return state

def hr_agent(state):
    resume_text = state["resume_text"]
    soft_skills = re.findall(r"(leadership|communication|teamwork|initiative|adaptability|problem-solving)", resume_text, re.I)
    desired_soft_skills = re.findall(r"(leadership|communication|teamwork|initiative|adaptability|problem-solving)", state["job_desc"], re.I)
    score = list_cosine_sim(desired_soft_skills, soft_skills)
    state["hr_score"] = float(score)
    return state

import re
import os
from dotenv import load_dotenv
from google import genai

# Load your API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "API key not found in .env!"

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.0-flash"

def feedback_agent(state):
    job_desc = state.get("job_desc", "")[:1000]
    resume_text = state.get("resume_text", "")[:3000]

    # Regex extraction
    skill_pattern = r"(Python|Java|C\+\+|SQL|Excel|Machine Learning|Data Science|React|Node\.js)"
    edu_pattern = r"(B\.?Tech|M\.?Tech|MBA|PhD|Bachelor|Master)"

    job_skills = set(re.findall(skill_pattern, job_desc, re.I))
    job_edu = set(re.findall(edu_pattern, job_desc, re.I))

    resume_skills = set(re.findall(skill_pattern, resume_text, re.I))
    resume_edu = set(re.findall(edu_pattern, resume_text, re.I))
    experience_years = re.findall(r"(\d+)\s+years?", resume_text)
    resume_exp = max([int(y) for y in experience_years], default=0)

    matched_skills = job_skills & resume_skills
    missing_skills = job_skills - resume_skills
    edu_match = bool(job_edu & resume_edu)
    missing_edu = job_edu - resume_edu

    feedback_lines = []
    feedback_lines.append(f"- Matched skills: {', '.join(matched_skills) if matched_skills else 'None'}")
    if missing_skills:
        feedback_lines.append(f"- Missing required skills: {', '.join(missing_skills)}")
    else:
        feedback_lines.append(f"- All required skills are present.")
    if edu_match:
        feedback_lines.append(f"- Education requirement met: {', '.join(job_edu & resume_edu)}")
    else:
        feedback_lines.append(f"- Missing required education: {', '.join(missing_edu) if missing_edu else 'Not specified'}")
    feedback_lines.append(f"- Experience: {resume_exp} years")

    # LLM summary prompt
    prompt = (
        "You are an expert resume reviewer.\n\n"
        "Given this job description:\n"
        f"{job_desc}\n\n"
        "And this resume:\n"
        f"{resume_text}\n\n"
        "Provide a brief, professional summary (1-2 sentences) with actionable feedback for improvement."
    )

    # Try Gemini, fallback to FLAN-T5 if Gemini fails
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        summary = response.candidates[0].content.parts[0].text.strip()
        feedback_lines.append(f"- LLM feedback summary: {summary}")
    except Exception as e:
        # Fallback to FLAN-T5
        flan_input = f"Summarize feedback for this resume and job description:\nJob: {job_desc}\nResume: {resume_text}"
        inputs = flan_tokenizer(flan_input, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = flan_model.generate(**inputs, max_new_tokens=64)
        summary = flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        feedback_lines.append(f"- FLAN-T5 feedback summary: {summary}")

    state["feedback"] = "\n".join(feedback_lines)
    return state



# Example usage:
# topn_resumes = [{"resume_text": "..."} for ...]
# job_desc = "..."
# feedbacks = generate_feedback_for_topn(topn_resumes, job_desc)
# for f in feedbacks:
#     print(f["feedback"])


# ---- GRAPH ----
def build_graph():
    graph = StateGraph(dict)
    graph.add_node("recruiter_agent", recruiter_agent)
    graph.add_node("analyst_agent", analyst_agent)
    graph.add_node("skills_agent", skills_agent)
    graph.add_node("education_agent", education_agent)
    graph.add_node("hr_agent", hr_agent)
    graph.add_node("feedback_agent", feedback_agent)

    graph.add_edge(START, "recruiter_agent")
    graph.add_edge("recruiter_agent", "analyst_agent")
    graph.add_edge("analyst_agent", "skills_agent")
    graph.add_edge("skills_agent", "education_agent")
    graph.add_edge("education_agent", "hr_agent")
    graph.add_edge("hr_agent", "feedback_agent")
    graph.add_edge("feedback_agent", END)
    return graph.compile()

graph = build_graph()

# ---- ROUTES ----
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/home.html")
def home_html():
    return render_template("home.html")

@app.route("/shortlist_page")
def shortlist_page():
    return render_template("index.html")

@app.route("/shortlist", methods=["POST"])
def shortlist():
    job_desc = request.form.get("job_description", "")
    top_n = int(request.form.get("top_n", 3))
    files = request.files.getlist("resumes")

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            resume_text = extract_resume_text(path)
            state = {"job_desc": job_desc, "resume_text": resume_text}
            result = graph.invoke(state)

            total = sum([
                float(result.get("analyst_score", 0)),
                float(result.get("skills_score", 0)),
                float(result.get("education_score", 0)),
                float(result.get("hr_score", 0))
            ]) / 4

            results.append({
                "filename": filename,
                "resume_text": resume_text,
                "scores": {
                    "analyst": float(result.get("analyst_score", 0)),
                    "skills": float(result.get("skills_score", 0)),
                    "education": float(result.get("education_score", 0)),
                    "hr": float(result.get("hr_score", 0))
                },
                "final_score": float(round(total, 2)),
                "feedback": result.get("feedback", "")
            })

    sorted_resumes = sorted(results, key=lambda x: x["final_score"], reverse=True)
    for idx, r in enumerate(sorted_resumes):
        if idx >= top_n:
            r["feedback"] = ""
    return jsonify({"resumes": sorted_resumes})

@app.route("/download_top_n", methods=["POST"])
def download_top_n():
    data = request.get_json()
    resumes = data.get("resumes", [])  # Only the selected resumes
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for r in resumes:
            filename = r["filename"]
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Only add the resume if it exists
            if os.path.exists(resume_path):
                zf.write(resume_path, arcname=filename)
            # Add feedback as a text file (write actual content, not empty)
            feedback_content = r.get("feedback", "").strip()
            feedback_filename = f"{os.path.splitext(filename)[0]}_feedback.txt"
            # Only add feedback file if feedback is not empty
            if feedback_content:
                zf.writestr(feedback_filename, feedback_content)
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='selected_resumes_with_feedback.zip'
    )


if __name__ == "__main__":
    app.run(debug=True)


