# ResumeAI — Resume Category Classification System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![ML Pipeline](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Containerized](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**GitHub Repository**: [https://github.com/pavankalyanbommi-wq/Resume_category](https://github.com/pavankalyanbommi-wq/Resume_category)

---

## 1. Project Overview

In recruitment and talent management workflows, technical recruiters and automated applicant tracking systems (ATS) receive large volumes of resumes in unstructured formats (PDF, DOCX, raw text). Manually screening and tagging candidate resumes across engineering and business domains is slow, inconsistent, and error-prone.

**ResumeAI** is an end-to-end Machine Learning Operations (MLOps) web application and REST API service that automates resume screening and domain classification. The system extracts text from documents, normalizes and redacts personal identifiable information (PII), extracts domain-specific technical skills, and classifies the profile into target industry disciplines with calibrated posterior probabilities and ambiguity detection.

---

## 2. Key Features

- **Document Ingestion & Sanitization**:
  - Secure parsing of PDF (`PyMuPDF`) and Word documents (`python-docx`).
  - Text preprocessing: Unicode normalization, whitespace cleanup, and PII masking (emails, phone numbers).
  - Validation safeguards for file size (5 MB limit), magic-bytes, and empty or corrupted payloads.

- **Machine Learning Inference Pipeline**:
  - TF-IDF vectorization with n-gram range `(1, 2)`, sublinear term frequency scaling, and custom resume boilerplate stop words.
  - Multi-class classification via regularized Logistic Regression.
  - 10 supported industry domains: Data Science, Python Developer, Java Developer, DevOps Engineer, Web Designing, Software Engineer, HR, Business Analyst, Database Administrator, and Network Security Engineer.

- **Ambiguity & Low-Confidence Detection**:
  - Real-time confidence calibration flagging multi-disciplinary profiles with confidence < 22% or margin between top 2 categories < 4%.
  - Keyword and skill extraction across 24+ technical domains.

- **Web Workspace & Dashboard**:
  - Responsive web interface featuring authentication, dashboard metrics, drag-and-drop document upload, and audit reports.
  - Session-based security, CSRF protection, and SQLite audit logging with SQLAlchemy 2.0.

- **Production-Ready REST API**:
  - Programmatic endpoints supporting both JSON text payloads and multipart document uploads.
  - Real-time `/health` check endpoint for uptime monitoring and container probes.

- **Cloud & Containerization**:
  - Non-root `Dockerfile` and `docker-compose.yml` for multi-platform deployment.
  - Ready for cloud hosting (AWS EC2 / GCP Compute) with Gunicorn WSGI.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Machine Learning** | `scikit-learn 1.5.2`, `pandas 2.2.3`, `numpy 2.0.2`, `joblib 1.4.2` |
| **Backend & Web Framework** | Python 3.10 / 3.11, `Flask 3.0.3`, `Werkzeug 3.0.3`, `Flask-WTF 1.2.1` |
| **Database & ORM** | SQLite 3, `SQLAlchemy 2.0.36` |
| **Document Processing** | `PyMuPDF 1.24.11` (fitz), `python-docx 1.1.2` |
| **Server & Deployment** | `Gunicorn 23.0.0`, `Docker`, `docker-compose` |
| **Testing & Verification** | `pytest 8.3.3`, `pytest-cov 5.0.0` |

---

## 4. Project Directory Structure

```text
Resume_category/
├── data/
│   ├── README.md               # Dataset documentation & guidelines
│   └── sample_resumes.csv      # Balanced dataset for 10 classification categories
├── database/
│   ├── db.py                   # Engine initialization, scoped sessions & migrations
│   └── models.py               # User and Prediction SQLAlchemy data models
├── instance/
│   └── resume_ai.db            # Persistent SQLite database (local instance)
├── model/
│   ├── evaluation_metrics.json # Test metrics, confusion matrix & classification report
│   └── resume_classifier.joblib# Serialized TF-IDF + Logistic Regression model pipeline
├── routes/
│   ├── api.py                  # Programmatic REST API endpoints (/api/predict)
│   ├── auth.py                 # User authentication (register, login, logout)
│   └── main.py                 # Core UI routes (landing, upload, prediction, reports, history, /health)
├── services/
│   ├── prediction_service.py   # Model loading, inference, confidence analysis & keyword extraction
│   └── resume_parser.py        # PDF & DOCX text extraction, validation, and PII redaction
├── static/
│   ├── css/
│   │   └── style.css           # Styling, responsive layout & themes
│   └── js/
│       └── app.js              # Interactive drag-and-drop file upload & UI scripts
├── templates/
│   ├── 404.html                # Error page
│   ├── base.html               # Base Jinja2 layout
│   ├── dashboard.html          # User dashboard and prediction statistics
│   ├── history.html            # User prediction history
│   ├── index.html              # Landing page
│   ├── login.html              # Authentication sign-in
│   ├── profile.html            # User profile overview
│   ├── register.html           # Authentication sign-up
│   ├── report.html             # Detailed printable classification report
│   ├── result.html             # Real-time classification results and probability distributions
│   └── upload.html             # Resume file upload and direct text input
├── tests/
│   ├── conftest.py             # Pytest configuration and in-memory test fixtures
│   ├── test_api.py             # REST API endpoint tests
│   ├── test_app.py             # Application flow and view tests
│   ├── test_auth.py            # User authentication & permission tests
│   └── test_model.py           # Model loading, vectorization, and inference tests
├── .dockerignore               # Docker exclusions
├── .env.example                # Example environment variable template
├── .gitignore                  # Git tracking rules
├── app.py                      # Flask application factory and entry point
├── config.py                   # Development, testing, and production configuration
├── docker-compose.yml          # Container compose specification
├── Dockerfile                  # Production container definition
├── pytest.ini                  # Pytest runner settings
├── README.md                   # Project documentation
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development and testing dependencies
└── train_model.py              # Model training, cross-validation, and export script
```

---

## 5. Local Setup & Installation

### Prerequisites

- **Python**: 3.10 or 3.11
- **Git**: Installed and configured on your system

### 1. Clone the Repository

```bash
git clone https://github.com/pavankalyanbommi-wq/Resume_category.git
cd Resume_category
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Optional: install development & testing packages
pip install -r requirements-dev.txt
```

### 4. Configure Environment Variables

Create a `.env` file from the provided `.env.example`:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS:**
```bash
cp .env.example .env
```

Default settings in `.env`:
```ini
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=dev-secret-key-replace-in-production-with-strong-random-token
DATABASE_URL=sqlite:///resume_ai.db
MODEL_PATH=model/resume_classifier.joblib
MAX_CONTENT_LENGTH=5242880
UPLOAD_FOLDER=uploads
PORT=5000
HOST=0.0.0.0
```

---

## 6. Running the Application

### Start the Flask Server

```bash
python app.py
```

The application will start and listen on `http://127.0.0.1:5000`:
- **Web UI**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Health Check**: [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)
- **Upload / Text Inference**: [http://127.0.0.1:5000/prediction](http://127.0.0.1:5000/prediction)
- **Dashboard**: [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard)

---

## 7. REST API Endpoints

### 7.1 Health Check

Verifies server status and model loading readiness.

- **Method**: `GET`
- **URL**: `/health`
- **Response** (`200 OK`):

```json
{
  "status": "ok",
  "service": "resume-category-classifier",
  "model_loaded": true,
  "timestamp": "2026-10-25T17:39:19.426470Z"
}
```

### 7.2 Resume Prediction Endpoint

Classifies resume text or uploaded documents.

- **Method**: `POST`
- **URL**: `/api/predict`
- **Accepted Content Types**: `application/json` or `multipart/form-data`

#### Option 1: Direct JSON Request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Senior Python Developer with 6 years experience in Flask, Django, PostgreSQL, Docker, AWS, and Machine Learning pipelines."
  }'
```

#### Option 2: File Upload (PDF or DOCX)

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@/path/to/resume.pdf"
```

#### Example JSON Response (`200 OK`)

```json
{
  "status": "success",
  "filename": "direct_text_input",
  "predicted_category": "Python Developer",
  "confidence": 0.2397,
  "confidence_percentage": 24.0,
  "is_low_confidence": false,
  "confidence_warning": null,
  "margin": 0.1065,
  "probabilities": [
    { "category": "Python Developer", "percentage": 24.0, "probability": 0.2397 },
    { "category": "Data Science", "percentage": 13.3, "probability": 0.1332 },
    { "category": "Software Engineer", "percentage": 11.1, "probability": 0.1111 },
    { "category": "Java Developer", "percentage": 9.1, "probability": 0.0907 },
    { "category": "DevOps Engineer", "percentage": 8.3, "probability": 0.0833 },
    { "category": "HR", "percentage": 8.1, "probability": 0.0807 },
    { "category": "Database Administrator", "percentage": 6.8, "probability": 0.068 },
    { "category": "Business Analyst", "percentage": 6.6, "probability": 0.0656 },
    { "category": "Web Designing", "percentage": 6.5, "probability": 0.0645 },
    { "category": "Network Security Engineer", "percentage": 6.3, "probability": 0.0632 }
  ],
  "top_keywords": [
    "Python",
    "Django",
    "PostgreSQL",
    "Docker",
    "Flask",
    "Machine Learning"
  ],
  "text_length": 122,
  "timestamp": "2026-10-25T17:39:34.438555Z"
}
```

---

## 8. Model Training & Re-training

To retrain the model and regenerate the serialized pipeline and metrics:

```bash
python train_model.py --dataset data/sample_resumes.csv --output model/resume_classifier.joblib --metrics-path model/evaluation_metrics.json
```

---

## 9. Docker Deployment

### Build and Run with Docker

```bash
docker build -t resumeai:latest .
docker run -d -p 5000:5000 --name resumeai-container resumeai:latest
```

### Run with Docker Compose

```bash
docker compose up -d
```

---

## 10. Automated Testing

Run the automated test suite with pytest:

```bash
pytest -v
```

Run test suite with coverage report:

```bash
pytest --cov=. --cov-report=term-missing
```

---

## 11. License

This project is licensed under the MIT License.
