from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
import fitz  # PyMuPDF
import spacy
import re
import zipfile
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="JobCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("../frontend/index.html")

nlp = spacy.load("en_core_web_sm")

ADZUNA_APP_ID = "4f890e3e"
ADZUNA_APP_KEY = "0708313fcfb2b40737d354aaa8e0b804"

# Pakistan is not supported by Adzuna — we search 'gb' (UK) as proxy for global remote jobs
COUNTRY_MAP = {
    "pk": "gb",   # Pakistan → search UK/global remote
    "gb": "gb",
    "us": "us",
    "au": "au",
    "ca": "ca",
    "de": "de",
    "in": "in",
    "sg": "sg",
}

SKILLS = [
    "python","javascript","typescript","java","c++","c#","go","rust","swift","kotlin","php","ruby","scala","r","matlab","sql","html","css",
    "machine learning","deep learning","neural network","nlp","computer vision","scikit-learn","tensorflow","pytorch","keras","pandas","numpy",
    "matplotlib","seaborn","opencv","langchain","transformers","bert","gpt","llm","rag","chromadb","faiss","hugging face",
    "react","next.js","vue","angular","node.js","express","django","flask","fastapi","spring","laravel","tailwind","bootstrap","graphql","rest api","websocket",
    "mysql","postgresql","mongodb","redis","elasticsearch","cassandra","firebase","supabase","sqlite",
    "data analysis","data visualization","tableau","power bi","etl","apache spark","hadoop","airflow",
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","github actions","ci/cd","linux","nginx","microservices","serverless","vercel","railway",
    "arduino","esp32","raspberry pi","iot","embedded systems","modbus","mqtt",
    "agile","scrum","git","unit testing","api","oop","data structures","algorithms","system design","tdd","rest","oauth","jwt",
]

# ── TEXT EXTRACTION ──
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from .docx by reading the XML inside the zip"""
    text = ""
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                for elem in root.iter(ns + "t"):
                    if elem.text:
                        text += elem.text + " "
    except Exception as e:
        raise HTTPException(400, f"Could not read Word file: {str(e)}")
    return text.lower()

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(400, "Please upload a PDF or Word (.docx) file")

# ── NLP HELPERS ──
def extract_skills(text: str) -> list:
    found = []
    text_lower = text.lower()
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))

def compute_match_score(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(score) * 100, 1)
    except:
        return 0.0

def get_experience_level(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["10+ years", "senior", "lead", "principal", "director", "head of"]):
        return "Senior"
    elif any(w in text_lower for w in ["5+ years", "3+ years", "mid", "intermediate"]):
        return "Mid-Level"
    else:
        return "Junior / Entry-Level"

def get_verdict(score: float, missing_count: int) -> dict:
    if score >= 70 and missing_count <= 2:
        return {"label": "Apply Now", "color": "#22c55e", "icon": "🚀", "message": "Your profile is a strong match. Apply with confidence."}
    elif score >= 50:
        return {"label": "Apply with Cover Letter", "color": "#f59e0b", "icon": "✍️", "message": "Good match but address skill gaps in your cover letter."}
    else:
        return {"label": "Upskill First", "color": "#ef4444", "icon": "📚", "message": "Build the missing skills before applying to maximize your chances."}

def generate_tips(missing: list, score: float, gaps: list) -> list:
    tips = []
    if missing:
        tips.append(f"Add these skills to your resume if you have them: {', '.join(missing[:3])}")
    if gaps:
        tips.append(f"Use these keywords from the job posting in your resume: {', '.join(gaps[:3])}")
    if score < 50:
        tips.append("Tailor your resume summary to mirror the job description language")
    if score >= 70:
        tips.append("You're a strong match — write a confident cover letter highlighting your relevant projects")
    tips.append("Quantify your achievements with numbers and metrics where possible")
    return tips[:4]

def match_jobs_to_resume(resume_skills: list, jobs: list) -> list:
    results = []
    for job in jobs:
        jd_text = (job.get("title", "") + " " + job.get("description", "")).lower()
        jd_skills = extract_skills(jd_text)
        matched = [s for s in resume_skills if s in jd_skills]
        missing = [s for s in jd_skills if s not in resume_skills][:5]
        total = len(jd_skills) if jd_skills else 1
        score = round((len(matched) / total) * 100, 1) if total > 0 else 50.0
        results.append({
            "title": job.get("title", "N/A"),
            "company": job.get("company", {}).get("display_name", "N/A"),
            "location": job.get("location", {}).get("display_name", "N/A"),
            "salary": f"${job.get('salary_min', 0):,.0f} – ${job.get('salary_max', 0):,.0f}" if job.get("salary_min") else "Not specified",
            "url": job.get("redirect_url", "#"),
            "match_score": score,
            "matched_skills": matched[:6],
            "missing_skills": missing,
            "verdict": get_verdict(score, len(missing)),
            "description": job.get("description", "")[:200] + "..." if job.get("description") else "",
        })
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results

# ── ENDPOINT 1: Analyze resume vs job description ──
@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    resume_bytes = await resume.read()
    resume_text = extract_text(resume_bytes, resume.filename)

    if not resume_text.strip():
        raise HTTPException(400, "Could not extract text from file")

    jd_text = job_description.lower()
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    matched_skills = [s for s in resume_skills if s in jd_skills]
    missing_skills = [s for s in jd_skills if s not in resume_skills]

    score = compute_match_score(resume_text, jd_text)
    if jd_skills:
        skill_score = (len(matched_skills) / len(jd_skills)) * 100
        final_score = round((score * 0.4) + (skill_score * 0.6), 1)
    else:
        final_score = score

    exp_level = get_experience_level(resume_text)
    verdict = get_verdict(final_score, len(missing_skills))

    doc = nlp(job_description)
    jd_keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 3]
    resume_words = set(resume_text.split())
    keyword_gaps = list(set([w for w in jd_keywords if w not in resume_words]))[:8]

    return {
        "match_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "keyword_gaps": keyword_gaps,
        "experience_level": exp_level,
        "verdict": verdict,
        "resume_skill_count": len(resume_skills),
        "jd_skill_count": len(jd_skills),
        "tips": generate_tips(missing_skills, final_score, keyword_gaps),
    }

# ── ENDPOINT 2: Find jobs from resume ──
@app.post("/find-jobs")
async def find_jobs(
    resume: UploadFile = File(...),
    country: str = Form("gb"),
    keywords: str = Form("")
):
    resume_bytes = await resume.read()
    resume_text = extract_text(resume_bytes, resume.filename)
    resume_skills = extract_skills(resume_text)

    search_query = keywords if keywords.strip() else " ".join(resume_skills[:4])
    if not search_query.strip():
        search_query = "software developer"

    # Map country code (Pakistan uses gb endpoint)
    adzuna_country = COUNTRY_MAP.get(country, "gb")
    is_pakistan = country == "pk"

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(
                f"https://api.adzuna.com/v1/api/jobs/{adzuna_country}/search/1",
                params={
                    "app_id": ADZUNA_APP_ID,
                    "app_key": ADZUNA_APP_KEY,
                    "results_per_page": 10,
                    "what": search_query,
                    "content-type": "application/json",
                }
            )
            data = resp.json()
            jobs = data.get("results", [])
        except Exception as e:
            raise HTTPException(502, f"Could not reach job listings API: {str(e)}")

    if not jobs:
        raise HTTPException(404, f"No jobs found for '{search_query}'. Try different keywords.")

    ranked = match_jobs_to_resume(resume_skills, jobs)

    return {
        "resume_skills": resume_skills,
        "search_query": search_query,
        "total_found": len(ranked),
        "pakistan_note": is_pakistan,
        "jobs": ranked,
    }
