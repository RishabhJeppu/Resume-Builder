import sys
import os

sys.path.append("./")
from llm.Groq_setup import Groq
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse


app = FastAPI()

UPLOAD_DIR = "./utils/uploads"
HTML_TEMPLATES_DIR = "./html_templates"

app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(HTML_TEMPLATES_DIR, "index.html"), "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.get("/next_page", response_class=HTMLResponse)
async def read_next_page():
    with open(os.path.join(HTML_TEMPLATES_DIR, "next_page.html"), "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.get("/preview_page", response_class=HTMLResponse)
async def preview_template():
    with open(os.path.join(HTML_TEMPLATES_DIR, "preview_page.html"), "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.post("/upload")
async def upload_resume(
    resumeFile: UploadFile = File(...),
    jobDescription: str = Form(...),
    templateType: str = Form(...),
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        file_path = os.path.join(UPLOAD_DIR, resumeFile.filename)
        with open(file_path, "wb") as file:
            content = await resumeFile.read()
            file.write(content)

        llm = Groq()
        llm.generate_json_resume(jobDescription)

        return RedirectResponse(url="/success", status_code=303)

    except Exception as e:
        # Handle errors and log them
        print(f"Error saving file: {e}")
        return HTMLResponse(
            content="An error occurred during file upload.", status_code=500
        )


@app.get("/success", response_class=HTMLResponse)
async def success_page():
    with open(os.path.join(HTML_TEMPLATES_DIR, "success.html"), "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.post("/process")
async def process_resume(file_path: str):
    """
    Process the uploaded resume file into structured JSON.
    """
    pass


@app.post("/enhance")
async def enhance_resume(resume_data: dict, api_key: str = Form(...)):
    """
    Use LLM to enhance the resume content.
    """
    pass


@app.post("/render")
async def render_resume(
    resume_data: dict,
    template_name: str = Form(...),
    section_order: list[str] = Form([]),
):
    """
    Render the tailored resume into LaTeX and generate a PDF.
    """
    pass


@app.get("/download/{file_name}")
async def download_resume(file_name: str):
    """
    Download the rendered resume (PDF, LaTeX, or JSON).
    """
    pass


@app.post("/generate_message")
async def generate_message(
    name: str = Form(...),
    job_title: str = Form(...),
    company: str = Form(...),
    skills: list[str] = Form(...),
    api_key: str = Form(...),
):
    """
    Generate a recruiter email or LinkedIn message using LLM.
    """
    pass
