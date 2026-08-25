from __future__ import annotations
import os, re
from copy import deepcopy
import streamlit as st
from cv_engine import RESUME_TEMPLATE, ProviderError, ats_analysis, build_docx, extract_document, run_pipeline

st.set_page_config(page_title="CV Studio", page_icon="📄", layout="wide")

def secret(name):
    try: return str(st.secrets.get(name, ""))
    except Exception: return os.environ.get(name, "")

for key, value in {"resume": deepcopy(RESUME_TEMPLATE), "job_text": ""}.items():
    if key not in st.session_state: st.session_state[key] = value

def update_collection(section, title, fields, bullet_key=None, remove_index=None):
    """Persist structural edits before Streamlit rebuilds the editor widgets."""
    items = st.session_state.resume.setdefault(section, [])
    if remove_index is None:
        item = {key: "" for key, _ in fields}
        if bullet_key: item[bullet_key] = []
        items.append(item)
    elif 0 <= remove_index < len(items):
        items.pop(remove_index)
    for key in list(st.session_state):
        if key.startswith(title + "_"): del st.session_state[key]

def collection(section, title, items, fields, bullet_key=None):
    st.markdown("#### " + title)
    result = []
    for index, item in enumerate(items):
        label = item.get(fields[0][0]) or "{} {}".format(title.rstrip("s"), index + 1)
        with st.expander(str(label), expanded=index == 0):
            current = dict(item); columns = st.columns(2)
            for position, (key, display) in enumerate(fields):
                current[key] = columns[position % 2].text_input(display, str(current.get(key, "")), key="{}_{}_{}".format(title, index, key))
            if bullet_key:
                lines = st.text_area("Bullet points — one per line", "\n".join(map(str, current.get(bullet_key, []))), height=120, key="{}_{}_bullets".format(title, index))
                current[bullet_key] = [line.strip().lstrip("-•").strip() for line in lines.splitlines() if line.strip()]
            st.button("Remove", key="remove_{}_{}".format(title, index), on_click=update_collection, args=(section, title, fields, bullet_key, index))
            result.append(current)
    st.button("+ Add " + title.rstrip("s"), key="add_" + title, on_click=update_collection, args=(section, title, fields, bullet_key))
    return result

st.markdown("""<style>.block-container{max-width:1280px;padding-top:1.4rem}.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#172554,#2563eb);color:white;margin-bottom:1rem}.hero h1{margin:0}.hero p{margin:.35rem 0 0;opacity:.9}[data-testid="stMetric"]{border:1px solid #e2e8f0;padding:12px;border-radius:12px;background:#f8fafc}</style><div class="hero"><h1>CV Studio</h1><p>Create a job-targeted, editable resume from your experience and target job description.</p></div>""", unsafe_allow_html=True)

# Credentials stay server-side and are never rendered in the interface.
openai_key = secret("OPENAI_API_KEY")
hf_token = secret("HF_TOKEN")
openai_model = "gpt-4o"
hf_model = "Qwen/Qwen2.5-72B-Instruct:fastest"

create, edit, export = st.tabs(["1 · Create", "2 · Edit", "3 · ATS & export"])
with create:
    left, right = st.columns(2)
    with left:
        st.subheader("Existing resume")
        resume_file = st.file_uploader("DOCX, PDF, or TXT", type=["docx", "pdf", "txt"], key="resume_file")
    with right:
        st.subheader("Target job")
        job_file = st.file_uploader("Optional JD document", type=["docx", "pdf", "txt"], key="job_file")
        pasted_job = st.text_area("Or paste the job description", st.session_state.job_text, height=155)
    st.markdown("**Workflow:** Analyze your experience and target job, then generate a tailored resume.")
    if st.button("✨ Create tailored resume", type="primary", use_container_width=True):
        if not resume_file: st.error("Upload a resume first.")
        elif not openai_key or not hf_token: st.error("Server API credentials are not configured. Add them to .streamlit/secrets.toml and restart the app.")
        else:
            try:
                source = extract_document(resume_file); job = extract_document(job_file) if job_file else pasted_job.strip()
                if not job: raise ValueError("Upload or paste a job description.")
                with st.spinner("Analyzing your documents and tailoring your resume…"):
                    _, tailored = run_pipeline(source, job, openai_key, hf_token, openai_model, hf_model)
                st.session_state.resume = tailored; st.session_state.job_text = job
                st.success("Your tailored resume is ready. Review all content in Edit.")
            except (ValueError, ProviderError) as error: st.error(str(error))

with edit:
    resume = st.session_state.resume
    st.info("Review all AI wording and remove any unsupported claim before applying.")
    columns = st.columns(2)
    fields = [("name", "Full name"), ("headline", "Headline"), ("email", "Email"), ("phone", "Phone"), ("location", "Location"), ("linkedin", "LinkedIn / portfolio")]
    for index, (key, display) in enumerate(fields): resume[key] = columns[index % 2].text_input(display, str(resume.get(key, "")), key="basic_" + key)
    resume["summary"] = st.text_area("Professional summary", str(resume.get("summary", "")), height=140)
    skills = st.text_area("Skills — comma-separated", ", ".join(map(str, resume.get("skills", []))))
    resume["skills"] = [item.strip() for item in skills.split(",") if item.strip()]
    resume["experience"] = collection("experience", "Experience", resume["experience"], [("role", "Role"), ("company", "Company"), ("location", "Location"), ("start_date", "Start"), ("end_date", "End")], "bullets")
    resume["projects"] = collection("projects", "Projects", resume["projects"], [("name", "Project"), ("technologies", "Technologies"), ("link", "Link")], "bullets")
    resume["education"] = collection("education", "Education", resume["education"], [("degree", "Degree"), ("institution", "Institution"), ("location", "Location"), ("start_date", "Start"), ("end_date", "End")], "details")
    resume["certifications"] = collection("certifications", "Certifications", resume["certifications"], [("name", "Certification"), ("issuer", "Issuer"), ("date", "Date")])
    st.session_state.resume = resume

with export:
    score, matched, missing = ats_analysis(st.session_state.resume, st.session_state.job_text)
    first, second, third = st.columns(3)
    first.metric("ATS guide", "{}%".format(score)); second.metric("Keywords present", len(matched)); third.metric("Keywords to review", len(missing))
    st.progress(score / 100.0)
    left, right = st.columns(2)
    left.success("Present: " + (", ".join(matched) or "None")); right.warning("Missing: " + (", ".join(missing[:20]) or "None"))
    st.caption("Keyword/completeness guidance only—not an employer ATS guarantee.")
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", st.session_state.resume.get("name") or "optimized_resume").strip("_")
    st.download_button("⬇ Download editable Word CV", build_docx(st.session_state.resume), filename + "_CV.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
