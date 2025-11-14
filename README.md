# RAG-Powered Clinical Decision Support System (CDSS)

A comprehensive Clinical Decision Support System powered by Retrieval-Augmented Generation (RAG), featuring advanced NLP processing, semantic analysis, and LLM-enhanced medical insights.

## 🏗️ Project Structure

```
TechBro_AB2_03-2/
├── cdss-react-frontend/          # React + Vite Frontend
│   ├── src/
│   │   ├── App.jsx               # Main application with chat interface
│   │   ├── components/           # Reusable UI components
│   │   │   ├── AnalysisPanel.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── Header.jsx
│   │   │   ├── MedicalSearch.jsx
│   │   │   └── ...
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── cdss_chatbot/                 # Django Backend with RAG System
│   ├── cdss_project/             # Django project settings
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── Rag/                      # Core RAG Application
│   │   ├── rag_system.py         # Main RAG implementation
│   │   ├── nlp_utils.py          # NLP preprocessing utilities
│   │   ├── llm_features.py       # LLM features (dense retrieval, summarization, QA)
│   │   ├── semantic_parser.py    # Medical semantic analysis
│   │   ├── enhanced_medical_system.py
│   │   ├── views.py              # API endpoints
│   │   ├── urls.py               # URL routing
│   │   ├── models.py             # Database models
│   │   ├── data/                 # Medical knowledge base
│   │   │   ├── comprehensive_top50_diseases_database.json
│   │   │   ├── pubmed_research_database.json
│   │   │   ├── comprehensive_drug_database.json
│   │   │   ├── clinical_guidelines_database.json
│   │   │   └── symptom_disease_lexicon.json
│   │   └── templates/            # HTML templates
│   ├── scripts/                  # Data processing scripts
│   ├── manage.py
│   ├── requirements.txt
│   └── venv/                     # Python virtual environment
│
└── README.md
```

## 🚀 Features

### Frontend (React)
- 💬 **Real-time Chat Interface** - Interactive chatbot with medical query processing
- 📊 **Detailed Analysis Panel** - Comprehensive display of NLP, RAG, and LLM results
- 🎨 **Modern UI** - Glassmorphism design with TailwindCSS
- ⚡ **Live Backend Status** - Real-time health monitoring
- 📱 **Responsive Design** - Works on desktop and mobile devices

### Backend (Django + RAG)
- 🧠 **Advanced NLP Processing**
  - Tokenization and sentence segmentation
  - Lemmatization and POS tagging
  - Spell correction for medical terms
  - Named Entity Recognition (NER)

- 🔍 **Dense Retrieval System**
  - PubMedBERT for biomedical text embeddings
  - FAISS vector database for fast similarity search
  - Semantic search capabilities

- 🤖 **LLM-Enhanced Features**
  - Google Gemini 2.0 Flash integration
  - Contextual response generation
  - Medical text summarization (extractive & abstractive)
  - Question answering system

- 📊 **Confidence Score & Risk Assessment** ⭐ NEW
  - AI confidence-based risk scoring for each diagnosis
  - Condition-specific risk level categorization
  - Automated alert generation for high-risk conditions
  - Multi-layered risk assessment combining confidence and severity
  - Real-time risk stratification (Critical/High/Medium/Low)

- 🏥 **Medical Knowledge Base**
  - 55+ diseases with comprehensive data
  - PubMed research papers integration
  - Drug database with medication information
  - Clinical guidelines and treatment protocols
  - Symptom-disease mapping

## 📋 Prerequisites

- **Node.js** 20.19.0 or higher (for frontend)
- **Python** 3.8+ (for backend)
- **Google Gemini API Key** (for LLM features)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd TechBro_AB2_03-2
```

### 2. Frontend Setup

```bash
cd cdss-react-frontend
npm install
```

### 3. Backend Setup

```bash
cd cdss_chatbot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK and spaCy data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

### 4. Configure API Key

Create or update `cdss_chatbot/cdss_project/settings.py`:

```python
GEMINI_API_KEY = "your-google-gemini-api-key-here"
```

## 🚀 Running the Application

### Start Backend (Django)

```bash
cd cdss_chatbot
venv\Scripts\activate  # Activate virtual environment
python manage.py migrate  # Run database migrations
python manage.py runserver  # Start on http://127.0.0.1:8000
```

### Start Frontend (React)

```bash
cd cdss-react-frontend
npm run dev  # Start on http://localhost:5173
```

### Access the Application

1. Open your browser and navigate to: **http://localhost:5173**
2. The frontend will automatically connect to the Django backend
3. Start chatting with the AI-powered CDSS!

## 🔗 API Endpoints

### Main Endpoints

- `GET /api/health/` - Backend health check
- `POST /api/rag-chat/` - Process medical queries with RAG
- `GET /api/test-all-features/` - Test all NLP, RAG, and LLM features
- `GET /api/patients/` - List patients
- `POST /api/patients/` - Create patient
- `POST /api/medical-knowledge-search/` - Search medical knowledge
- `POST /api/risk-assessment/` - Perform risk assessment

## 🧬 How RAG Works

### 1. Query Processing
```
User Query → NLP Preprocessing → Entity Extraction → Query Enhancement
```

### 2. Retrieval Phase
```
Enhanced Query → Dense Retrieval (PubMedBERT) → FAISS Search → Top-K Documents
```

### 3. Generation Phase
```
Retrieved Context + Original Query → LLM (Gemini) → Enhanced Response
```

### 4. Analysis Output
- Differential diagnoses with confidence scores
- **Confidence-based risk assessment with automated alerts**
- Treatment recommendations
- Multi-layered risk assessment (overall + condition-specific)
- Research paper citations
- Clinical guidelines
- Follow-up suggestions

## 🎯 Confidence Score Feature

The CDSS includes an advanced confidence score system that provides risk assessment based on AI diagnostic confidence:

### Key Features:
- **Confidence-Based Risk Scoring**: Combines AI confidence (0-100%) with condition-specific risk levels
- **Automated Alerts**: 
  - 🚨 **Critical Alerts** for high-risk conditions (e.g., Heart Attack, Stroke)
  - ⚠️ **High Risk Alerts** for urgent conditions (e.g., Pneumonia)
- **Multi-Tier Risk Levels**:
  - **Critical** (≥80%): Immediate emergency attention required
  - **High** (60-79%): Urgent medical evaluation needed
  - **Medium** (40-59%): Medical consultation recommended
  - **Low** (<40%): Routine follow-up suggested

### Example Output:
```json
{
  "risk_assessment": {
    "confidence_based_risk": {
      "Heart Attack": {
        "risk_score": 0.900,
        "risk_level": "Critical",
        "confidence": 90.0
      }
    },
    "alerts": [
      "🚨 CRITICAL ALERT: Heart Attack detected with 90.0% confidence - Seek immediate emergency medical attention!"
    ]
  }
}
```

📖 **Full Documentation**: See [CONFIDENCE_SCORE_FEATURE.md](cdss_chatbot/CONFIDENCE_SCORE_FEATURE.md) for detailed information.

🧪 **Test the Feature**: Run `python cdss_chatbot/test_confidence_score.py` to see it in action.

## 📊 Medical Knowledge Base

The system uses multiple JSON databases:

- **55+ Diseases** covering 11 categories:
  - Cardiovascular (Heart Disease, Stroke, Hypertension, etc.)
  - Respiratory (COPD, Asthma, Pneumonia, etc.)
  - Endocrine/Metabolic (Diabetes, Obesity, etc.)
  - Neurological (Alzheimer's, Parkinson's, etc.)
  - Mental Health (Depression, Anxiety, etc.)
  - And more...

- **PubMed Research Papers** - Evidence-based medical literature
- **Drug Database** - Comprehensive medication information
- **Clinical Guidelines** - Treatment protocols and best practices

## 🔧 Troubleshooting

### Frontend Issues

**Vite module errors:**
```bash
cd cdss-react-frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Backend Issues

**Missing dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Database errors:**
```bash
python manage.py migrate
```

## 📚 Technology Stack

### Frontend
- React 19
- Vite 7
- TailwindCSS 4
- Axios for API calls

### Backend
- Django 4.2+
- Python 3.8+
- NLTK & spaCy (NLP)
- Sentence Transformers (Embeddings)
- FAISS (Vector Search)
- Google Gemini API (LLM)
- PubMedBERT (Biomedical Embeddings)

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**VaidyaSathi** - Your AI-Powered Clinical Decision Support Companion 🏥🤖
