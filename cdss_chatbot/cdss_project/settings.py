import os
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-your-secret-key-here-change-in-production')
if SECRET_KEY == 'django-insecure-your-secret-key-here-change-in-production':
    print("⚠️  WARNING: Using default SECRET_KEY! Please set SECRET_KEY in your .env file")
    print("⚠️  Generate a new key with: python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '127.0.0.1,localhost,testserver').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',  # Add CORS headers
    'Rag',  # Add our app here
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add CORS middleware first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS settings for development
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite React frontend
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
]

ROOT_URLCONF = 'cdss_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'cdss_project.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Encoding settings
DEFAULT_CHARSET = 'utf-8'
FILE_CHARSET = 'utf-8'

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "Rag" / "static",
]

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================
# AI and API Configuration
# ============================================

# Google Gemini API Key (REQUIRED - from environment variable)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment variables!")
    print("⚠️  Please set GEMINI_API_KEY in your .env file")
    print("⚠️  Get your API key from: https://makersuite.google.com/app/apikey")
    # For development only - allow startup without key but features won't work
    if DEBUG:
        print("⚠️  Running in DEBUG mode without API key - AI features will be limited")
        GEMINI_API_KEY = None
    else:
        raise ValueError("GEMINI_API_KEY environment variable is required in production")

# Pinecone API Configuration (Optional - for vector database)
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST = os.getenv('PINECONE_HOST')
if PINECONE_API_KEY and PINECONE_HOST:
    print("✅ Pinecone configuration loaded from environment variables")
else:
    print("ℹ️  Pinecone not configured - using local FAISS vector database")

# Medical Knowledge Base Path
MEDICAL_KNOWLEDGE_PATH = BASE_DIR / os.getenv('MEDICAL_KNOWLEDGE_PATH', 'Rag/data/')

# ============================================
# NLP and Semantic Parsing Configuration
# ============================================
ENABLE_SEMANTIC_PARSING = os.getenv('ENABLE_SEMANTIC_PARSING', 'True').lower() == 'true'
SCISPACY_MODEL = os.getenv('SCISPACY_MODEL', 'en_core_sci_sm')
UMLS_LINKING = os.getenv('UMLS_LINKING', 'True').lower() == 'true'

# ============================================
# Optional AI Model Overrides
# ============================================
DENSE_RETRIEVAL_MODEL = os.getenv('DENSE_RETRIEVAL_MODEL', 'pritamdeka/S-PubMedBert-MS-MARCO')
QA_MODEL = os.getenv('QA_MODEL', 'deepset/roberta-base-squad2')
SUMMARIZATION_MODEL = os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn')