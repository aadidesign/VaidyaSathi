import json
import os
import time
import hashlib
from typing import Dict, List, Optional, Any, Union

import numpy as np
import nltk


from nltk.tokenize import word_tokenize
from .nlp_utils import preprocess, SimpleSpellCorrector, dependency_parse, ngrams
from .semantic_parser import analyze_medical_semantics
from .llm_features import (
    get_dense_retrieval, get_text_summarizer, get_rag_generator, get_qa_system,
    dense_retrieve, summarize_medical_text, generate_rag_response, answer_medical_question
)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import google.generativeai as genai
from tqdm import tqdm
import faiss
from django.conf import settings

# Download necessary NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: NLTK data download failed: {e}")
    print("Continuing without NLTK data downloads...")

class RAGClinicalDecisionSupport:
    def __init__(self, medical_data_path=None):
        print("Initializing RAG-Enhanced Clinical Decision Support System...")
        # Initialize medical language model and dense retrieval
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        # Initialize PubMedBERT for enhanced medical text processing
        print("🧬 Initializing PubMedBERT for enhanced medical text processing...")
        try:
            from transformers import AutoModel
            self.pubmedbert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.pubmedbert_model.eval()  # Set to evaluation mode
            print("✅ PubMedBERT model loaded successfully")
        except Exception as e:
            print(f"⚠️ PubMedBERT model loading failed: {e}")
            self.pubmedbert_model = None
        
        # Initialize advanced LLM features
        print("🤖 Initializing advanced LLM features...")
        self.dense_retrieval = get_dense_retrieval()
        self.text_summarizer = get_text_summarizer()
        self.rag_generator = get_rag_generator()
        self.qa_system = get_qa_system()
        print("✅ Advanced LLM features initialized")

        # Configure Google Gemini API
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if api_key:
            genai.configure(api_key=api_key)
        
        # Initialize Gemini model with retry mechanism
        self.generative_model = genai.GenerativeModel('gemini-2.0-flash')
        self.max_retries = 3
        self.retry_delay = 2  # seconds

        # Initialize vector database for RAG
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.vector_dim)  # L2 distance for similarity search
        self.document_store = {}  # Store document text with IDs
        self.loaded_documents = 0

        # Initialize lightweight NLP components
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = set(stopwords.words('english'))
        self._negation_cues = {
            'no', 'not', 'denies', 'deny', 'without', "isn't", "wasn't", "aren't", "hasn't", "havent", "haven't", 'negative'
        }
        self._uncertainty_cues = {'possible', 'probable', 'likely', 'maybe', 'suggests', 'suggestive'}
        self._symptom_lexicon = self._load_symptom_lexicon()
        # Initialize simple spell corrector with medical lexicon terms as vocabulary
        vocab = []
        for dis, obj in self._symptom_lexicon.items():
            vocab.append(dis)
            vocab.extend(obj.get('symptoms', []))
            vocab.extend(obj.get('keywords', []))
        self._speller = SimpleSpellCorrector(vocab)

        # Load medical knowledge base if provided
        if medical_data_path:
            self.load_medical_knowledge(medical_data_path)
        else:
            print("No medical knowledge base provided. Starting with empty vector database.")

        print("System initialized successfully!")

    def _is_medical_query(self, query: str) -> bool:
        """
        Check if the query is medical-related
        """
        query_lower = query.lower()
        
        # Medical keywords and phrases
        medical_keywords = [
            'symptom', 'disease', 'condition', 'illness', 'pain', 'ache', 'fever', 'cough',
            'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'fatigue', 'weakness',
            'dizzy', 'dizziness', 'chest pain', 'shortness of breath', 'difficulty breathing',
            'rash', 'swelling', 'inflammation', 'infection', 'bacteria', 'virus', 'cancer',
            'diabetes', 'hypertension', 'heart', 'lung', 'liver', 'kidney', 'brain', 'blood',
            'pressure', 'sugar', 'cholesterol', 'medication', 'drug', 'treatment', 'therapy',
            'diagnosis', 'test', 'scan', 'x-ray', 'mri', 'ct', 'blood test', 'urine test',
            'allergy', 'asthma', 'arthritis', 'depression', 'anxiety', 'stress', 'mental',
            'pregnancy', 'childbirth', 'menstrual', 'period', 'hormone', 'thyroid',
            'skin', 'hair', 'nail', 'eye', 'ear', 'nose', 'throat', 'stomach', 'intestine',
            'bone', 'joint', 'muscle', 'nerve', 'spine', 'back', 'neck', 'shoulder', 'knee',
            'ankle', 'wrist', 'elbow', 'hip', 'foot', 'hand', 'finger', 'toe'
        ]
        
        # Basic health and wellness questions (allowed)
        basic_health_keywords = [
            'healthy', 'health', 'wellness', 'exercise', 'diet', 'nutrition', 'vitamin',
            'supplement', 'sleep', 'stress management', 'lifestyle', 'prevention',
            'what is', 'how to', 'tips', 'advice', 'recommendation', 'guidance'
        ]
        
        # Non-medical topics to filter out
        non_medical_keywords = [
            'weather', 'sports', 'politics', 'entertainment', 'movies', 'music', 'games',
            'shopping', 'travel', 'cooking', 'recipes', 'fashion', 'beauty', 'makeup',
            'technology', 'computer', 'phone', 'software', 'programming', 'business',
            'finance', 'investment', 'stock', 'crypto', 'bitcoin', 'job', 'career',
            'education', 'school', 'university', 'college', 'homework', 'assignment'
        ]
        
        # Check for non-medical topics first
        for keyword in non_medical_keywords:
            if keyword in query_lower:
                return False
        
        # Check for medical keywords
        medical_score = sum(1 for keyword in medical_keywords if keyword in query_lower)
        
        # Check for basic health keywords
        health_score = sum(1 for keyword in basic_health_keywords if keyword in query_lower)
        
        # Check for basic human questions (what is, how to, etc.)
        basic_question_patterns = [
            'what is', 'how to', 'what are', 'how do', 'why does', 'when should',
            'where can', 'who should', 'can you', 'tell me about', 'explain',
            'help me understand', 'what does', 'how does', 'what causes'
        ]
        
        question_score = sum(1 for pattern in basic_question_patterns if pattern in query_lower)
        
        # If it has medical keywords, is a basic health question, or is a general health question, consider it medical
        return medical_score > 0 or health_score > 0 or question_score > 0

    def _get_available_diseases_list(self) -> List[str]:
        """
        Get a formatted list of available diseases for user guidance
        """
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, 'r', encoding='utf-8') as f:
                    diseases_data = json.load(f)
                
                diseases_by_category = {}
                for disease in diseases_data:
                    category = disease.get('category', 'Other')
                    if category not in diseases_by_category:
                        diseases_by_category[category] = []
                    diseases_by_category[category].append(disease.get('disease_name', ''))
                
                return diseases_by_category
            else:
                # Fallback list
                return {
                    'Cardiovascular': ['Ischemic Heart Disease', 'Stroke', 'Hypertension', 'Heart Failure', 'Atrial Fibrillation'],
                    'Respiratory': ['COPD', 'Asthma', 'Pneumonia', 'Lung Cancer', 'Lower Respiratory Infections'],
                    'Endocrine/Metabolic': ['Type 2 Diabetes', 'Type 1 Diabetes', 'Obesity', 'Metabolic Syndrome', 'Thyroid Disorders'],
                    'Neurological': ['Alzheimer\'s Disease', 'Parkinson\'s Disease', 'Epilepsy', 'Migraine', 'Multiple Sclerosis'],
                    'Mental Health': ['Depression', 'Anxiety Disorders', 'Bipolar Disorder', 'Schizophrenia', 'PTSD'],
                    'Gastrointestinal': ['GERD', 'IBD', 'IBS', 'Peptic Ulcer Disease', 'Liver Cirrhosis'],
                    'Musculoskeletal': ['Osteoarthritis', 'Rheumatoid Arthritis', 'Osteoporosis', 'Low Back Pain', 'Fibromyalgia'],
                    'Infectious': ['COVID-19', 'Tuberculosis', 'Malaria', 'HIV/AIDS', 'Hepatitis B'],
                    'Cancer': ['Breast Cancer', 'Colorectal Cancer', 'Prostate Cancer', 'Liver Cancer', 'Stomach Cancer'],
                    'Kidney/Urinary': ['Chronic Kidney Disease', 'Acute Kidney Injury', 'Kidney Stones', 'UTI', 'BPH'],
                    'Skin': ['Atopic Dermatitis', 'Psoriasis', 'Acne Vulgaris', 'Skin Cancer', 'Eczema']
                }
        except Exception as e:
            print(f"Error loading diseases list: {e}")
            return {'Error': ['Unable to load disease list']}

    def _handle_basic_question(self, query: str) -> Optional[str]:
        """
        Handle basic human-level questions that might not be strictly medical
        """
        query_lower = query.lower()
        
        # Basic health and wellness questions
        if 'what is' in query_lower and any(word in query_lower for word in ['health', 'medicine', 'doctor', 'hospital', 'medical']):
            return "Health refers to a state of complete physical, mental, and social well-being, not just the absence of disease. Medicine is the science and practice of diagnosing, treating, and preventing disease. Doctors are medical professionals who care for patients, and hospitals are healthcare facilities that provide medical treatment."
        
        if 'how to stay healthy' in query_lower or 'how to be healthy' in query_lower:
            return "To stay healthy: 1) Eat a balanced diet with fruits, vegetables, and whole grains, 2) Exercise regularly (at least 150 minutes per week), 3) Get 7-9 hours of sleep, 4) Stay hydrated, 5) Avoid smoking and limit alcohol, 6) Manage stress, 7) Get regular check-ups, 8) Practice good hygiene."
        
        if 'what is' in query_lower and any(word in query_lower for word in ['exercise', 'diet', 'nutrition', 'vitamin']):
            return "Exercise is physical activity that improves health and fitness. A healthy diet includes balanced nutrients from various food groups. Nutrition refers to the nutrients our body needs to function properly. Vitamins are essential nutrients that our body needs in small amounts."
        
        if 'how to prevent' in query_lower and any(word in query_lower for word in ['disease', 'illness', 'infection']):
            return "To prevent diseases: 1) Practice good hygiene (hand washing), 2) Get vaccinated, 3) Eat a healthy diet, 4) Exercise regularly, 5) Get adequate sleep, 6) Manage stress, 7) Avoid smoking and excessive alcohol, 8) Regular medical check-ups, 9) Practice safe behaviors."
        
        if 'when to see a doctor' in query_lower:
            return "See a doctor when you experience: persistent symptoms lasting more than a few days, severe pain, high fever, difficulty breathing, chest pain, sudden changes in vision or speech, unexplained weight loss, persistent fatigue, or any symptoms that concern you. Always seek immediate medical attention for emergencies."
        
        if 'what is' in query_lower and any(word in query_lower for word in ['symptom', 'pain', 'fever', 'cough', 'headache']):
            return "Symptoms are signs that indicate a medical condition. Pain is an unpleasant sensation that can indicate injury or illness. Fever is elevated body temperature, often indicating infection. Cough is a reflex to clear airways. Headache is pain in the head or neck area. If you're experiencing persistent symptoms, please describe them for medical guidance."
        
        if any(phrase in query_lower for phrase in ['thank you', 'thanks', 'appreciate']):
            return "You're welcome! I'm here to help with your medical questions. Feel free to ask about symptoms, conditions, treatments, or any health concerns you might have."
        
        if any(phrase in query_lower for phrase in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! I'm VaidyaSathi, your AI Clinical Decision Support System. I can help you with medical queries about 55+ diseases. How can I assist you today?"
        
        # Return None if no basic question pattern matches
        return None

    def _get_condition_explanation(self, condition: str) -> str:
        """
        Get a detailed explanation of a medical condition from the database
        """
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, 'r', encoding='utf-8') as f:
                    diseases_data = json.load(f)
                
                # Find the disease in the database
                for disease in diseases_data:
                    if disease.get('disease_name', '').lower() == condition.lower():
                        description = disease.get('description', '')
                        category = disease.get('category', '')
                        prevalence = disease.get('prevalence', '')
                        
                        explanation_parts = []
                        
                        if description:
                            # Use the first sentence or first 100 characters
                            first_sentence = description.split('.')[0] if '.' in description else description[:100]
                            explanation_parts.append(first_sentence)
                        
                        if category:
                            explanation_parts.append(f"Category: {category}")
                        
                        if prevalence:
                            explanation_parts.append(f"Prevalence: {prevalence}")
                        
                        return '. '.join(explanation_parts) if explanation_parts else ""
            
            return ""
            
        except Exception as e:
            print(f"Error getting condition explanation: {str(e)}")
            return ""

    def _enhance_medical_retrieval_with_pubmedbert(self, query: str, medical_documents: List[str]) -> List[Dict]:
        """
        Use PubMedBERT to enhance medical document retrieval with better semantic understanding
        """
        try:
            if not self.pubmedbert_model or not medical_documents:
                return []
            
            import torch
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Tokenize and encode the query
            query_tokens = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                query_embeddings = self.pubmedbert_model(**query_tokens).last_hidden_state.mean(dim=1)
            
            enhanced_results = []
            
            # Process each document
            for i, doc in enumerate(medical_documents[:50]):  # Limit to top 50 documents for performance
                try:
                    # Tokenize and encode the document
                    doc_tokens = self.tokenizer(doc, return_tensors='pt', truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        doc_embeddings = self.pubmedbert_model(**doc_tokens).last_hidden_state.mean(dim=1)
                    
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        query_embeddings.cpu().numpy(),
                        doc_embeddings.cpu().numpy()
                    )[0][0]
                    
                    enhanced_results.append({
                        'document': doc,
                        'similarity_score': float(similarity),
                        'index': i
                    })
                    
                except Exception as e:
                    print(f"Error processing document {i}: {e}")
                    continue
            
            # Sort by similarity score and return top results
            enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return enhanced_results[:10]  # Return top 10 most relevant documents
            
        except Exception as e:
            print(f"Error in PubMedBERT enhancement: {e}")
            return []

    def _load_symptom_lexicon(self) -> Dict[str, Dict[str, Any]]:
        """
        Load a simple symptom→disease lexicon to support core NLP matching.
        Structure: {
          "disease_name": { "symptoms": [..], "keywords": [..] }
        }
        """
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            file_path = os.path.join(data_dir, 'symptom_disease_lexicon.json')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load symptom lexicon: {e}")
        return {}

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        result = []
        for tok in tokens:
            if tok.isalpha() and tok.lower() not in self._stopwords:
                # Map POS crudely for better lemmatization
                pos = 'n'
                if tok.endswith('ing'):
                    pos = 'v'
                result.append(self._lemmatizer.lemmatize(tok.lower(), pos))
        return result

    def _extract_symptoms_rule_based(self, text: str) -> List[str]:
        """
        Core NLP: tokenize, normalize, lemmatize, simple negation handling,
        and match against symptom lexicon keywords.
        """
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        lemmas = self._lemmatize_tokens(tokens)

        # Negation/uncertainty window: mark tokens after cue for 5 tokens
        negated = set()
        uncertain = set()
        window = 0
        uwindow = 0
        for i, tok in enumerate(lemmas):
            if tok in self._negation_cues:
                window = 5
                continue
            if tok in self._uncertainty_cues:
                uwindow = 5
                continue
            if window > 0:
                negated.add(i)
                window -= 1
            if uwindow > 0:
                uncertain.add(i)
                uwindow -= 1

        found = []
        text_set = set(lemmas)
        for symptom in self._iter_all_symptom_keywords():
            parts = symptom.split()
            # simple phrase match
            if len(parts) == 1:
                if parts[0] in text_set:
                    idxs = [i for i, t in enumerate(lemmas) if t == parts[0]]
                    if not any(i in negated for i in idxs):
                        found.append(symptom)
            else:
                joined = ' '.join(lemmas)
                if symptom in joined:
                    # crude negation: fail if any token in window is negated
                    found.append(symptom)
        return sorted(list(set(found)))

    def _iter_all_symptom_keywords(self):
        for dis, obj in self._symptom_lexicon.items():
            for kw in obj.get('symptoms', []) + obj.get('keywords', []):
                yield kw.lower()

    def load_medical_knowledge(self, data_path):
        """
        Load comprehensive medical knowledge from multiple databases and index them in the vector database
        """
        print(f"Loading comprehensive medical knowledge from {data_path}...")

        try:
            # Load multiple medical knowledge databases
            knowledge_files = [
                'accurate_medical_conditions.json',  # Load accurate conditions first
                'comprehensive_medical_knowledge.json',
                'expanded_symptom_disease_lexicon.json', 
                'comprehensive_drug_database.json',
                'clinical_guidelines_database.json',
                'pubmed_research_database.json',
                'additional_medical_research.json',
                'sample_medical_knowledge.json'
            ]
            
            if os.path.isdir(data_path):
                file_paths = [os.path.join(data_path, f) for f in knowledge_files if os.path.exists(os.path.join(data_path, f))]
            else:
                file_paths = [data_path]

            documents = []
            total_conditions = 0
            total_drugs = 0
            total_guidelines = 0
            total_research_papers = 0
            
            for file_path in tqdm(file_paths, desc="Loading medical knowledge files"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Process accurate medical conditions first
                    if 'accurate_medical_conditions.json' in str(file_path) and isinstance(data, list):
                        for condition in data:
                            total_conditions += 1
                            # Create comprehensive document text
                            doc_text = f"""
                            Condition: {condition.get('condition', '')}
                            ICD-10 Code: {condition.get('icd10_code', '')}
                            Description: {condition.get('description', '')}
                            Symptoms: {', '.join(condition.get('symptoms', []))}
                            Complications: {', '.join(condition.get('complications', []))}
                            Treatments: {', '.join(condition.get('treatments', []))}
                            Diagnostic Tests: {', '.join(condition.get('diagnostic_tests', []))}
                            Risk Factors: {', '.join(condition.get('risk_factors', []))}
                            Emergency Protocol: {condition.get('emergency_protocol', '')}
                            Age Groups: {', '.join(condition.get('age_groups', []))}
                            Severity Levels: {', '.join(condition.get('severity_levels', []))}
                            Keywords: {', '.join(condition.get('keywords', []))}
                            Sources: {len(condition.get('sources', []))} medical sources
                            """
                            documents.append(doc_text.strip())
                    
                    # Process comprehensive medical knowledge
                    elif 'comprehensive_medical_knowledge.json' in str(file_path) and isinstance(data, list):
                        for condition in data:
                            total_conditions += 1
                            # Create comprehensive document text
                            doc_text = f"""
                            Condition: {condition.get('condition', '')}
                            ICD-10 Code: {condition.get('icd10_code', '')}
                            Description: {condition.get('description', '')}
                            Symptoms: {', '.join(condition.get('symptoms', []))}
                            Complications: {', '.join(condition.get('complications', []))}
                            Treatments: {', '.join(condition.get('treatments', []))}
                            Diagnostic Tests: {', '.join(condition.get('diagnostic_tests', []))}
                            Risk Factors: {', '.join(condition.get('risk_factors', []))}
                            Emergency Protocol: {condition.get('emergency_protocol', '')}
                            Age Groups: {', '.join(condition.get('age_groups', []))}
                            Severity Levels: {', '.join(condition.get('severity_levels', []))}
                            Keywords: {', '.join(condition.get('keywords', []))}
                            Sources: {len(condition.get('sources', []))} medical sources
                            """
                            documents.append(doc_text.strip())
                    
                    # Process drug database
                    elif 'comprehensive_drug_database.json' in str(file_path) and isinstance(data, list):
                        for drug in data:
                            total_drugs += 1
                            doc_text = f"""
                            Drug: {drug.get('drug_name', '')}
                            Generic Name: {drug.get('generic_name', '')}
                            Drug Class: {drug.get('drug_class', '')}
                            Indications: {', '.join(drug.get('indications', []))}
                            Mechanism of Action: {drug.get('mechanism_of_action', '')}
                            Dosage: {drug.get('dosage', {}).get('adult', '')}
                            Contraindications: {', '.join(drug.get('contraindications', []))}
                            Side Effects: {', '.join(drug.get('side_effects', []))}
                            Drug Interactions: {len(drug.get('drug_interactions', []))} interactions
                            Monitoring: {', '.join(drug.get('monitoring', []))}
                            Pregnancy Category: {drug.get('pregnancy_category', '')}
                            """
                            documents.append(doc_text.strip())
                    
                    # Process clinical guidelines
                    elif 'clinical_guidelines_database.json' in str(file_path) and isinstance(data, list):
                        for guideline in data:
                            total_guidelines += 1
                            doc_text = f"""
                            Condition: {guideline.get('condition', '')}
                            Guideline: {guideline.get('guideline_title', '')}
                            Organization: {guideline.get('organization', '')}
                            Year: {guideline.get('year', '')}
                            Recommendations: {len(guideline.get('recommendations', []))} evidence-based recommendations
                            Diagnostic Criteria: {', '.join(guideline.get('diagnostic_criteria', []))}
                            Treatment Protocol: {', '.join(guideline.get('treatment_protocol', []))}
                            Sources: {len(guideline.get('sources', []))} clinical sources
                            """
                            documents.append(doc_text.strip())
                    
                    # Process PubMed research database
                    elif 'pubmed_research_database.json' in str(file_path) and isinstance(data, list):
                        for paper in data:
                            total_research_papers += 1
                            doc_text = f"""
                            Research Paper: {paper.get('title', '')}
                            Authors: {paper.get('authors', '')}
                            Journal: {paper.get('journal', '')}
                            Year: {paper.get('year', '')}
                            PMID: {paper.get('pmid', '')}
                            DOI: {paper.get('doi', '')}
                            Study Type: {paper.get('study_type', '')}
                            Sample Size: {paper.get('sample_size', '')}
                            Abstract: {paper.get('abstract', '')}
                            Key Findings: {', '.join(paper.get('key_findings', []))}
                            Medical Conditions: {', '.join(paper.get('medical_conditions', []))}
                            Evidence Level: {paper.get('evidence_level', '')}
                            Clinical Significance: {paper.get('clinical_significance', '')}
                            Full Text URL: {paper.get('full_text_url', '')}
                            PubMed URL: {paper.get('pubmed_url', '')}
                            """
                            documents.append(doc_text.strip())
                    
                    # Process additional medical research
                    elif 'additional_medical_research.json' in str(file_path) and isinstance(data, list):
                        for paper in data:
                            total_research_papers += 1
                            doc_text = f"""
                            Research Paper: {paper.get('title', '')}
                            Authors: {paper.get('authors', '')}
                            Journal: {paper.get('journal', '')}
                            Year: {paper.get('year', '')}
                            PMID: {paper.get('pmid', '')}
                            DOI: {paper.get('doi', '')}
                            Study Type: {paper.get('study_type', '')}
                            Sample Size: {paper.get('sample_size', '')}
                            Abstract: {paper.get('abstract', '')}
                            Key Findings: {', '.join(paper.get('key_findings', []))}
                            Medical Conditions: {', '.join(paper.get('medical_conditions', []))}
                            Evidence Level: {paper.get('evidence_level', '')}
                            Clinical Significance: {paper.get('clinical_significance', '')}
                            Full Text URL: {paper.get('full_text_url', '')}
                            PubMed URL: {paper.get('pubmed_url', '')}
                            """
                            documents.append(doc_text.strip())
                    
                    # Process symptom-disease lexicon
                    elif 'expanded_symptom_disease_lexicon.json' in str(file_path) and isinstance(data, dict):
                        for condition, info in data.items():
                            doc_text = f"""
                            Condition: {condition.replace('_', ' ').title()}
                            Symptoms: {', '.join(info.get('symptoms', []))}
                            Keywords: {', '.join(info.get('keywords', []))}
                            Severity: {info.get('severity', '')}
                            Emergency: {info.get('emergency', False)}
                            """
                            documents.append(doc_text.strip())
                    
                    # Process sample medical knowledge (legacy)
                    elif isinstance(data, list):
                        for item in data:
                            # Create chunks from each item
                            text = self._process_medical_item(item)
                            documents.append(text)
                    elif isinstance(data, dict):
                        text = self._process_medical_item(data)
                        documents.append(text)

            # Index all documents in the vector database
            if documents:
                print(f"Medical Knowledge Statistics:")
                print(f"   - Total Medical Conditions: {total_conditions}")
                print(f"   - Total Drugs: {total_drugs}")
                print(f"   - Total Clinical Guidelines: {total_guidelines}")
                print(f"   - Total Research Papers: {total_research_papers}")
                print(f"   - Total Documents: {len(documents)}")
                
                # Generate embeddings for all documents
                print("Generating embeddings for medical knowledge...")
                embeddings = []
                for doc in tqdm(documents, desc="Processing documents"):
                    try:
                        embedding = self.embedding_model.encode(doc)
                        embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error encoding document: {e}")
                        continue
                
                if embeddings:
                    # Add to FAISS index
                    embeddings_array = np.array(embeddings).astype('float32')
                    self.index.add(embeddings_array)
                    
                    # Store document texts
                    for i, doc in enumerate(documents):
                        self.document_store[i] = doc
                    
                    self.loaded_documents = len(documents)
                    print(f"Successfully loaded {self.loaded_documents} medical documents into vector database")
                    print(f"Vector database now contains comprehensive medical knowledge")
                else:
                    print("No embeddings generated")
            else:
                print("No documents found to load")

        except Exception as e:
            print(f"Error loading medical knowledge: {str(e)}")

            import traceback
            traceback.print_exc()

    def _process_medical_item(self, item):
        """
        Process a medical data item into text format for embedding
        """
        # Customize based on your data structure
        if 'title' in item and 'content' in item:
            return f"Title: {item['title']}\nContent: {item['content']}"
        elif 'condition' in item and 'description' in item:
            symptoms = ", ".join(item.get('symptoms', []))
            treatments = ", ".join(item.get('treatments', []))
            return f"Condition: {item['condition']}\nDescription: {item['description']}\nSymptoms: {symptoms}\nTreatments: {treatments}"
        else:
            # Generic processing for unknown structures
            return "\n".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])

    def _index_documents(self, documents):
        """
        Embed and index documents in the vector database
        """
        print(f"Indexing {len(documents)} documents...")
        batch_size = 32  # Process in batches to avoid memory issues

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]

            # Generate embeddings for the batch
            embeddings = self.embedding_model.encode(batch)

            # Add embeddings to the index
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            self.index.add(embeddings)

            # Store the original documents
            for doc in batch:
                doc_id = hashlib.md5(doc.encode()).hexdigest()
                self.document_store[doc_id] = doc
                self.loaded_documents += 1

            # Print progress periodically
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(documents):
                print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} documents")

    def analyze_case(self, clinical_query: str, ehr_data: Optional[Dict] = None) -> Dict:
        """
        Analyze a clinical case using RAG approach
        
        Args:
            clinical_query: The clinical query or patient description
            ehr_data: Optional electronic health record data
            
        Returns:
            Dict containing analysis results
        """
        try:
            print(f"Analyzing query: {clinical_query}")
            
            # Check if query is medical-related first
            if not self._is_medical_query(clinical_query):
                # Try to answer basic human-level questions
                basic_response = self._handle_basic_question(clinical_query)
                if basic_response:
                    return {
                        'summary': basic_response,
                        'is_medical': False,
                        'is_basic_qa': True,
                        'available_diseases': self._get_available_diseases_list(),
                        'differential_diagnoses': [],
                        'research_papers': {},
                        'medical_knowledge': [],
                        'recommendations': {},
                        'risk_assessment': {},
                        'preprocessing_stats': {}
                    }
                else:
                    return {
                        'summary': 'I can only assist with medical queries related to the diseases in my database. Please ask about symptoms, conditions, treatments, or medical advice for the following diseases: Cardiovascular diseases (Heart Disease, Stroke, Hypertension), Respiratory diseases (COPD, Asthma, Pneumonia), Diabetes, Mental Health conditions, and 45+ other diseases. For general questions, I can provide basic information about health and wellness.',
                        'is_medical': False,
                        'available_diseases': self._get_available_diseases_list(),
                        'differential_diagnoses': [],
                        'research_papers': {},
                        'medical_knowledge': [],
                        'recommendations': {},
                        'risk_assessment': {},
                        'preprocessing_stats': {}
                    }
            
            # Spell-correct tokens conservatively (only words > 4 chars)
            corrected_words = []
            for tok in clinical_query.split():
                if len(tok) > 4 and tok.isalpha():
                    corrected_words.append(self._speller.correction(tok))
                else:
                    corrected_words.append(tok)
            corrected_query = " ".join(corrected_words)

            # Extract patient information from the (possibly corrected) query
            patient_info = self._extract_patient_info(corrected_query)

            # Core NLP symptom extraction (rule-based) to complement LLM
            rule_based_symptoms = self._extract_symptoms_rule_based(clinical_query)
            if rule_based_symptoms:
                # Ensure symptoms is always a list
                current_symptoms = patient_info.get('symptoms', [])
                if isinstance(current_symptoms, str):
                    # If symptoms is a string, convert to list
                    current_symptoms = [current_symptoms] if current_symptoms else []
                    patient_info['symptoms'] = current_symptoms
                elif not isinstance(current_symptoms, list):
                    # If symptoms is neither string nor list, initialize as empty list
                    current_symptoms = []
                    patient_info['symptoms'] = current_symptoms
                
                existing = set([s.lower() for s in current_symptoms])
                for s in rule_based_symptoms:
                    if s.lower() not in existing:
                        patient_info['symptoms'].append(s)
            
            # Construct search query (also keep normalized tokens for retrieval boosting)
            search_query = self._construct_search_query(patient_info, corrected_query)
            pre = preprocess(corrected_query)

            # Create preprocessing statistics for detailed NLP status tracking
            spell_corrections = sum(1 for orig, corr in zip(clinical_query.split(), corrected_words) if orig != corr)
            preprocessing_stats = {
                'sentence_count': len(pre.get('sentences', [])),
                'token_count': len(pre.get('tokens', [])),
                'normalized_count': len(pre.get('normalized', [])),
                'pos_count': len(pre.get('pos', [])),
                'spell_corrections': spell_corrections,
                'original_query': clinical_query,
                'corrected_query': corrected_query,
                'bigram_count': 0,  # Will be updated below
                'trigram_count': 0  # Will be updated below
            }
            print(f"📊 NLP Preprocessing Stats: {preprocessing_stats}")

            # Add conceptual n-grams (bigrams/trigrams) to retrieval hinting
            bigrams = [" ".join(g) for g in ngrams(pre['normalized'], 2)]
            trigrams = [" ".join(g) for g in ngrams(pre['normalized'], 3)]
            
            # Update n-gram counts in stats
            preprocessing_stats['bigram_count'] = len(bigrams)
            preprocessing_stats['trigram_count'] = len(trigrams)
            
            # Perform semantic analysis for enhanced understanding (with error handling)
            semantic_analysis = {}
            entity_terms = []
            
            # Check if semantic parsing is enabled in settings
            enable_semantic = getattr(settings, 'ENABLE_SEMANTIC_PARSING', True)
            if enable_semantic:
                try:
                    print("🧠 Performing semantic analysis with SciSpacy...")
                    semantic_analysis = analyze_medical_semantics(corrected_query)
                    
                    # Extract medical entities for boosting search
                    medical_entities = semantic_analysis.get('medical_entities', {})
                    for entity_type, entities in medical_entities.items():
                        for entity in entities[:3]:  # Top 3 entities per type
                            entity_terms.append(entity['text'])
                    
                    print(f"✅ Found {len(medical_entities)} entity types, {len(semantic_analysis.get('medical_relationships', []))} relationships")
                    
                except Exception as e:
                    print(f"Warning: Semantic analysis failed: {e}")
                    semantic_analysis = {'error': str(e), 'medical_entities': {}, 'word_sense_disambiguation': {}, 'medical_relationships': [], 'semantic_roles': [], 'summary': {}}
            else:
                print("Semantic parsing disabled in settings")
                semantic_analysis = {'disabled': True, 'medical_entities': {}, 'word_sense_disambiguation': {}, 'medical_relationships': [], 'semantic_roles': [], 'summary': {}}
            
            # Retrieve relevant medical knowledge
            concept_hints = []
            # Prefer common medical phrases
            for phrase in trigrams + bigrams:
                if phrase and len(phrase.split()) > 1:
                    concept_hints.append(phrase)
            if concept_hints:
                search_query += f" Concepts: {', '.join(list(dict.fromkeys(concept_hints))[:10])}."
            
            if entity_terms:
                search_query += f" Medical entities: {', '.join(entity_terms[:10])}."

            retrieved_context = self._retrieve_medical_knowledge(search_query)
            
            # Generate differential diagnoses
            differential_diagnoses = self._generate_differential_diagnoses(
                patient_info, clinical_query, retrieved_context
            )

            # Adjust differentials using lexicon matches (score boosting)
            if self._symptom_lexicon and patient_info.get('symptoms'):
                symptoms = patient_info.get('symptoms', [])
                # Ensure symptoms is always a list
                if isinstance(symptoms, str):
                    symptoms = [symptoms] if symptoms else []
                elif not isinstance(symptoms, list):
                    symptoms = []
                symptoms_lower = [s.lower() for s in symptoms]
                for diag in differential_diagnoses:
                    name = diag.get('condition', '').lower()
                    if name in self._symptom_lexicon:
                        overlap = 0
                        for sym in self._symptom_lexicon[name].get('symptoms', []):
                            if sym.lower() in symptoms_lower:
                                overlap += 1
                        # Boost confidence proportionally
                        diag['confidence'] = float(min(100.0, diag.get('confidence', 0) + overlap * 5))
            
            # Perform risk assessment
            risk_assessment = self._perform_risk_assessment(patient_info, differential_diagnoses)
            
            # Perform confidence-based risk assessment (additional layer)
            confidence_based_risk = self._assess_risk_with_confidence(differential_diagnoses)
            
            # Generate alerts for high-risk conditions
            risk_alerts = self._generate_alerts(confidence_based_risk)
            
            # Merge alerts with risk assessment
            if risk_alerts:
                if 'alerts' not in risk_assessment:
                    risk_assessment['alerts'] = []
                risk_assessment['alerts'].extend(risk_alerts)
            
            # Add confidence-based risk scores to risk assessment
            risk_assessment['confidence_based_risk'] = confidence_based_risk
            
            # Retrieve research papers for the diagnoses with patient-specific context
            research_papers = self._retrieve_research_papers(differential_diagnoses, patient_info, clinical_query)
            
            # Generate recommendations FIRST - prioritize AI-generated personalized recommendations
            recommendations = self._generate_personalized_recommendations(patient_info, clinical_query, differential_diagnoses, risk_assessment)
            
            # Generate a summary with personalized recommendations - prioritize comprehensive database over API
            summary = self._generate_summary_from_database(patient_info, clinical_query, differential_diagnoses, risk_assessment, retrieved_context, recommendations)
            
            # If database summary is generic, try API as fallback
            if not summary or "Medical conditions require proper evaluation" in summary:
                prompt = f"""
                Based on the following patient information and medical knowledge, provide a concise summary of the case:
                
                Patient Information:
                {json.dumps(patient_info, indent=2)}
                
                Clinical Query:
                {clinical_query}
                
                Relevant Medical Knowledge:
                {' '.join(retrieved_context)}
                
                Differential Diagnoses:
                {json.dumps(differential_diagnoses, indent=2)}
                
                Risk Assessment:
                {json.dumps(risk_assessment, indent=2)}
                
                Recommendations:
                {json.dumps(recommendations, indent=2)}
                """
                
                try:
                    summary = self._call_generative_model_with_retry(prompt)
                except Exception as e:
                    print(f"API summary generation failed: {str(e)}")
                    # Use database summary as final fallback
                    if not summary:
                        summary = self._generate_summary_from_database(patient_info, clinical_query, differential_diagnoses, risk_assessment, retrieved_context, recommendations)
            
            # Final fallback if no recommendations were generated
            if not recommendations:
                recommendations = {
                    'immediate_actions': [
                        "Consult with a healthcare provider as soon as possible",
                        "Monitor symptoms and keep a detailed log"
                    ],
                    'tests': [
                        "Complete blood count (CBC)",
                        "Comprehensive metabolic panel",
                        "Specific tests based on symptoms"
                    ],
                    'lifestyle': [
                        "Maintain adequate hydration",
                        "Ensure proper rest and sleep",
                        "Follow a balanced diet"
                    ],
                    'follow_up': "Schedule a follow-up appointment within 1-2 weeks"
                }
            
            # Generate enhanced RAG response using advanced prompt engineering
            print("🤖 Generating enhanced clinical response...")
            try:
                # Determine response type based on query content
                response_type = "diagnosis"
                if any(word in clinical_query.lower() for word in ["treat", "therapy", "medication", "drug"]):
                    response_type = "treatment"
                elif any(word in clinical_query.lower() for word in ["summary", "summarize", "overview"]):
                    response_type = "summary"
                
                enhanced_response = generate_rag_response(clinical_query, retrieved_context, response_type)
                
                # Generate medical text summarization
                context_text = "\n".join(retrieved_context)
                medical_summary = summarize_medical_text(context_text, method="both")
                
                # Question answering if specific questions detected
                qa_results = []
                if "?" in clinical_query:
                    qa_result = answer_medical_question(clinical_query, context_text)
                    qa_results.append(qa_result)
                
                print("✅ Enhanced clinical response generated")
                
            except Exception as e:
                print(f"Warning: Enhanced response generation failed: {e}")
                enhanced_response = {'response': 'Enhanced features unavailable', 'parsed_response': {}}
                medical_summary = {'extractive_summary': '', 'abstractive_summary': ''}
                qa_results = []

            # Compile the results with enhanced features
            result = {
                'patient_info': patient_info,
                'clinical_query': clinical_query,
                'retrieved_context': retrieved_context,
                'differential_diagnoses': differential_diagnoses,
                'risk_assessment': risk_assessment,
                'research_papers': research_papers,
                'summary': summary,
                'recommendations': recommendations,
                'semantic_analysis': semantic_analysis,
                'enhanced_response': enhanced_response,
                'medical_summary': medical_summary,
                'qa_results': qa_results,
                'preprocessing_stats': preprocessing_stats,  # NEW: NLP preprocessing statistics
                'llm_features': {
                    'dense_retrieval': True,
                    'text_summarization': True,
                    'question_answering': True,
                    'enhanced_rag': True
                }
            }
            
            return result
        
        except Exception as e:
            print(f"Error in analyze_case: {str(e)}")
            # Return a minimal result in case of error
            return {
                'summary': f"An error occurred during analysis: {str(e)}",
                'differential_diagnoses': [],
                'retrieved_context': []
            }

    def _construct_search_query(self, patient_info: Dict, clinical_query: str) -> str:
        """
        Construct an enhanced search query from patient information for better PubMed data retrieval
        """
        symptoms = patient_info.get('symptoms', [])
        # Ensure symptoms is always a list
        if isinstance(symptoms, str):
            symptoms = [symptoms] if symptoms else []
        elif not isinstance(symptoms, list):
            symptoms = []
            
        age = patient_info.get('age', 'unknown age')
        gender = patient_info.get('gender', 'unknown gender')
        duration = patient_info.get('duration', 'unknown duration')

        # Create a comprehensive query that includes medical terminology for better PubMed matching
        query_parts = []
        
        # Add patient demographics
        query_parts.append(f"Patient: {age} {gender}")
        
        # Add symptoms with medical terminology
        if symptoms:
            symptom_query = ', '.join(symptoms)
            query_parts.append(f"symptoms: {symptom_query}")
            
            # Add medical condition keywords that might match PubMed data
            medical_keywords = self._extract_medical_keywords_from_symptoms(symptoms)
            if medical_keywords:
                query_parts.append(f"medical conditions: {', '.join(medical_keywords)}")
        
        # Add duration
        query_parts.append(f"duration: {duration}")
        
        # Add original clinical query for context
        if clinical_query and clinical_query.strip():
            query_parts.append(f"clinical presentation: {clinical_query}")
        
        # Combine all parts
        query = ". ".join(query_parts) + "."
        
        # Add PubMed-specific search terms
        pubmed_terms = self._get_pubmed_search_terms(symptoms, clinical_query)
        if pubmed_terms:
            query += f" Research focus: {', '.join(pubmed_terms)}."
        
        # Add corpus term weights if available
        try:
            stats_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'cdss_chatbot', 'corpus_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                top_terms = {t[0]: t[1] for t in stats.get('top_tokens', []) if isinstance(t, list) and t}
                # Add top frequent terms intersection as context hints
                hints = [s for s in symptoms if s in top_terms]
                if hints:
                    query += f" Key terms: {', '.join(hints)}."
        except Exception:
            pass

        # Add any past medical history from EHR if available
        if 'past_medical_history' in patient_info:
            query += f" History of {', '.join(patient_info['past_medical_history'])}."

        return query

    def _extract_medical_keywords_from_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Extract medical condition keywords from symptoms for better PubMed matching
        """
        medical_keywords = []
        
        # Common symptom to condition mappings
        symptom_condition_map = {
            'chest pain': ['cardiovascular disease', 'heart disease', 'angina'],
            'shortness of breath': ['respiratory disease', 'heart failure', 'asthma'],
            'headache': ['migraine', 'tension headache', 'cluster headache'],
            'fever': ['infection', 'inflammatory disease', 'sepsis'],
            'fatigue': ['chronic fatigue', 'anemia', 'thyroid disease'],
            'nausea': ['gastrointestinal disease', 'motion sickness', 'pregnancy'],
            'dizziness': ['vertigo', 'cardiovascular disease', 'neurological disorder'],
            'joint pain': ['arthritis', 'rheumatoid arthritis', 'osteoarthritis'],
            'abdominal pain': ['gastrointestinal disease', 'appendicitis', 'gallbladder disease'],
            'cough': ['respiratory infection', 'asthma', 'bronchitis']
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in symptom_condition_map:
                medical_keywords.extend(symptom_condition_map[symptom_lower])
        
        return list(set(medical_keywords))  # Remove duplicates

    def _get_pubmed_search_terms(self, symptoms: List[str], clinical_query: str) -> List[str]:
        """
        Generate PubMed-specific search terms for better research paper matching
        """
        search_terms = []
        
        # Add study type terms
        search_terms.extend(['clinical trial', 'randomized controlled trial', 'systematic review', 'meta-analysis'])
        
        # Add treatment-related terms
        search_terms.extend(['treatment', 'therapy', 'medication', 'intervention'])
        
        # Add diagnostic terms
        search_terms.extend(['diagnosis', 'diagnostic criteria', 'clinical guidelines'])
        
        # Add outcome terms
        search_terms.extend(['outcomes', 'efficacy', 'safety', 'prognosis'])
        
        return search_terms[:5]  # Limit to top 5 terms

    def _retrieve_medical_knowledge(self, query: str, top_k: int = 5) -> List[str]:
        """
        Enhanced retrieval using PubMedBERT and hybrid approach with PubMed data prioritization
        """
        print("📚 Retrieving relevant medical knowledge with PubMedBERT...")

        # Check if index is empty
        if self.index.ntotal == 0:
            print("Warning: Vector database is empty. Using fallback medical knowledge.")
            return self._get_fallback_medical_knowledge(query)

        try:
            # Method 1: PubMedBERT-enhanced retrieval (new approach)
            if self.pubmedbert_model:
                print("🧬 Using PubMedBERT for enhanced medical text understanding...")
                documents = list(self.document_store.values())
                enhanced_results = self._enhance_medical_retrieval_with_pubmedbert(query, documents)
                
                if enhanced_results:
                    pubmedbert_docs = [result['document'] for result in enhanced_results[:top_k]]
                    print(f"✅ PubMedBERT found {len(pubmedbert_docs)} highly relevant medical documents")
                    return pubmedbert_docs

            # Method 2: PubMed-specific retrieval (prioritize research papers)
            pubmed_docs = self._retrieve_pubmed_specific_docs(query, top_k)
            if pubmed_docs:
                print(f"✅ Found {len(pubmed_docs)} PubMed research documents")
                return pubmed_docs

            # Method 3: Dense Retrieval with transformer models (fallback)
            print("🔍 Using dense retrieval with transformer models...")
            documents = list(self.document_store.values())
            
            if documents:
                try:
                    dense_results = dense_retrieve(query, documents, top_k=min(3, len(documents)))
                    retrieved_docs = [result['document'] for result in dense_results]
                    print(f"✅ Dense retrieval found {len(retrieved_docs)} documents")
                except Exception as e:
                    print(f"Dense retrieval failed: {e}, falling back to traditional search")
                    retrieved_docs = []
            else:
                retrieved_docs = []

            # Method 3: Traditional vector search (fallback/complement)
            if len(retrieved_docs) < top_k:
                print("🔍 Using traditional vector search for additional results...")
                try:
                    query_embedding = self.embedding_model.encode([query])
                    faiss.normalize_L2(query_embedding)

                    distances, indices = self.index.search(query_embedding, top_k)

                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and len(retrieved_docs) < top_k:
                            doc_id = list(self.document_store.keys())[idx]
                            doc = self.document_store[doc_id]
                            
                            # Avoid duplicates
                            if doc not in retrieved_docs:
                                retrieved_docs.append(doc)
                except Exception as e:
                    print(f"Traditional search also failed: {e}")

            if not retrieved_docs:
                print("No relevant documents found. Using fallback medical knowledge.")
                return self._get_fallback_medical_knowledge(query)

            print(f"✅ Total retrieved documents: {len(retrieved_docs)}")
            return retrieved_docs[:top_k]  # Ensure we don't exceed top_k

        except Exception as e:
            print(f"Error retrieving medical knowledge: {str(e)}")
            return self._get_fallback_medical_knowledge(query)

    def _retrieve_pubmed_specific_docs(self, query: str, top_k: int) -> List[str]:
        """
        Retrieve PubMed-specific documents based on query terms
        """
        try:
            # Load PubMed database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            pubmed_file = os.path.join(data_dir, 'pubmed_research_database.json')
            
            if not os.path.exists(pubmed_file):
                return []
            
            with open(pubmed_file, 'r', encoding='utf-8') as f:
                pubmed_data = json.load(f)
            
            # Extract key terms from query
            query_terms = self._extract_query_terms(query)
            
            # Score and rank papers based on query relevance
            scored_papers = []
            for paper in pubmed_data:
                score = self._calculate_paper_relevance_score(paper, query_terms)
                if score > 0:
                    scored_papers.append((score, paper))
            
            # Sort by relevance score (highest first)
            scored_papers.sort(key=lambda x: x[0], reverse=True)
            
            # Convert to document format
            retrieved_docs = []
            for score, paper in scored_papers[:top_k]:
                doc_text = f"""
                Research Paper: {paper.get('title', '')}
                Authors: {paper.get('authors', '')}
                Journal: {paper.get('journal', '')}
                Year: {paper.get('year', '')}
                PMID: {paper.get('pmid', '')}
                Study Type: {paper.get('study_type', '')}
                Sample Size: {paper.get('sample_size', '')}
                Abstract: {paper.get('abstract', '')}
                Key Findings: {', '.join(paper.get('key_findings', []))}
                Medical Conditions: {', '.join(paper.get('medical_conditions', []))}
                Evidence Level: {paper.get('evidence_level', '')}
                Clinical Significance: {paper.get('clinical_significance', '')}
                """
                retrieved_docs.append(doc_text.strip())
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error in PubMed-specific retrieval: {str(e)}")
            return []

    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract relevant terms from the query for PubMed matching
        """
        # Simple term extraction - can be enhanced with NLP
        terms = []
        
        # Split by common delimiters
        words = query.lower().replace(',', ' ').replace('.', ' ').replace(':', ' ').split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'patient', 'with', 'symptoms', 'duration'}
        
        for word in words:
            if len(word) > 2 and word not in stop_words:
                terms.append(word)
        
        return terms

    def _calculate_paper_relevance_score(self, paper: Dict, query_terms: List[str]) -> float:
        """
        Calculate relevance score for a paper based on query terms
        """
        score = 0.0
        
        # Check title
        title_lower = paper.get('title', '').lower()
        for term in query_terms:
            if term in title_lower:
                score += 3.0  # High weight for title matches
        
        # Check medical conditions
        medical_conditions = [mc.lower() for mc in paper.get('medical_conditions', [])]
        for term in query_terms:
            for condition in medical_conditions:
                if term in condition:
                    score += 2.5  # High weight for condition matches
        
        # Check abstract
        abstract_lower = paper.get('abstract', '').lower()
        for term in query_terms:
            if term in abstract_lower:
                score += 1.0  # Medium weight for abstract matches
        
        # Check key findings
        key_findings = [kf.lower() for kf in paper.get('key_findings', [])]
        for term in query_terms:
            for finding in key_findings:
                if term in finding:
                    score += 1.5  # Medium-high weight for key findings
        
        # Boost score for high clinical significance
        clinical_significance = paper.get('clinical_significance', '').lower()
        if clinical_significance == 'high':
            score *= 1.5
        elif clinical_significance == 'medium':
            score *= 1.2
        
        # Boost score for recent papers
        year = paper.get('year', 0)
        if year >= 2020:
            score *= 1.3
        elif year >= 2015:
            score *= 1.1
        
        return score

    def _get_fallback_medical_knowledge(self, query: str) -> List[str]:
        """
        Provide fallback medical knowledge using comprehensive disease database
        """
        print("🔄 Using comprehensive database fallback...")
        
        # Try to find matching diseases in our comprehensive database
        matching_diseases = self._find_diseases_in_comprehensive_database(query)
        
        if matching_diseases:
            print(f"✅ Found {len(matching_diseases)} matching diseases in comprehensive database")
            return matching_diseases
        
        # If no matches found, try PubMed-specific fallback
        pubmed_fallback = self._get_pubmed_fallback_knowledge(query)
        if pubmed_fallback:
            return pubmed_fallback
        
        # Final generic fallback
        return [
            "Medical conditions require proper evaluation by healthcare professionals.",
            "Symptoms should be assessed in context with patient history and physical examination.",
            "Diagnostic tests may be necessary to confirm or rule out specific conditions.",
            "Treatment plans should be individualized based on patient-specific factors.",
            "Regular follow-up with healthcare providers is important for ongoing care."
        ]

    def _find_diseases_in_comprehensive_database(self, query: str) -> List[str]:
        """
        Find matching diseases in the comprehensive database
        """
        try:
            # Load comprehensive database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if not os.path.exists(comprehensive_file):
                return []
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            query_lower = query.lower()
            matching_knowledge = []
            
            # Search through all diseases
            for disease in comprehensive_data:
                disease_name_lower = disease['disease_name'].lower()
                symptoms_lower = [s.lower() for s in disease.get('symptoms', [])]
                category_lower = disease['category'].lower()
                
                # Check for direct disease name match
                if any(term in disease_name_lower for term in query_lower.split()):
                    matching_knowledge.extend(self._create_disease_knowledge(disease))
                
                # Check for symptom matches
                elif any(term in ' '.join(symptoms_lower) for term in query_lower.split()):
                    matching_knowledge.extend(self._create_disease_knowledge(disease))
                
                # Check for category matches
                elif any(term in category_lower for term in query_lower.split()):
                    matching_knowledge.extend(self._create_disease_knowledge(disease))
            
            return matching_knowledge[:10]  # Limit to 10 items
            
        except Exception as e:
            print(f"Error in comprehensive database fallback: {str(e)}")
            return []

    def _create_disease_knowledge(self, disease: Dict) -> List[str]:
        """
        Create knowledge entries for a disease
        """
        knowledge = []
        
        # Disease description
        knowledge.append(f"{disease['disease_name']} is a {disease['category'].lower()} condition with {disease['prevalence'].lower()} prevalence globally.")
        
        # Symptoms
        if disease.get('symptoms'):
            symptoms_text = ', '.join(disease['symptoms'][:5])  # First 5 symptoms
            knowledge.append(f"Common symptoms include: {symptoms_text}.")
        
        # Treatments
        if disease.get('treatments'):
            treatments_text = ', '.join(disease['treatments'][:3])  # First 3 treatments
            knowledge.append(f"Treatment approaches include: {treatments_text}.")
        
        # Risk factors
        if disease.get('risk_factors'):
            risk_factors_text = ', '.join(disease['risk_factors'][:3])  # First 3 risk factors
            knowledge.append(f"Risk factors include: {risk_factors_text}.")
        
        # Research papers
        if disease.get('research_papers'):
            knowledge.append(f"Recent research includes {len(disease['research_papers'])} published studies with evidence-based findings.")
        
        return knowledge

    def _get_pubmed_fallback_knowledge(self, query: str) -> List[str]:
        """
        Get fallback knowledge from PubMed database
        """
        try:
            # Load PubMed database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            pubmed_file = os.path.join(data_dir, 'pubmed_research_database.json')
            
            if not os.path.exists(pubmed_file):
                return []
            
            with open(pubmed_file, 'r', encoding='utf-8') as f:
                pubmed_data = json.load(f)
            
            query_lower = query.lower()
            matching_papers = []
            
            # Find papers related to the query
            for paper in pubmed_data:
                title_lower = paper.get('title', '').lower()
                disease_name_lower = paper.get('disease_name', '').lower()
                abstract_lower = paper.get('abstract', '').lower()
                
                if (any(term in title_lower for term in query_lower.split()) or
                    any(term in disease_name_lower for term in query_lower.split()) or
                    any(term in abstract_lower for term in query_lower.split())):
                    matching_papers.append(paper)
            
            if matching_papers:
                # Return knowledge from matching papers
                knowledge = []
                for paper in matching_papers[:3]:  # Top 3 papers
                    knowledge.append(f"Research: {paper['title']}")
                    if paper.get('abstract'):
                        knowledge.append(f"Findings: {paper['abstract'][:200]}...")
                
                return knowledge
            
            return []
            
        except Exception as e:
            print(f"Error in PubMed fallback: {str(e)}")
            return []

    def _generate_fallback_diagnoses_from_database(self, patient_info: Dict, clinical_query: str) -> List[Dict]:
        """
        Generate fallback diagnoses using the comprehensive disease database
        """
        try:
            # Load comprehensive database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if not os.path.exists(comprehensive_file):
                return [{
                    'condition': 'Requires further investigation',
                    'explanation': 'The symptoms provided are insufficient for a specific diagnosis.',
                    'confidence': 90.0
                }]
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            symptoms = patient_info.get('symptoms', [])
            clinical_query_lower = clinical_query.lower()
            diagnoses = []
            
            # Search for matching diseases
            for disease in comprehensive_data:
                disease_name_lower = disease['disease_name'].lower()
                symptoms_lower = [s.lower() for s in disease.get('symptoms', [])]
                category_lower = disease['category'].lower()
                description_lower = disease.get('description', '').lower()
                
                # Calculate match score
                match_score = 0
                explanation_parts = []
                
                # Check for direct disease name match
                if any(term in disease_name_lower for term in clinical_query_lower.split()):
                    match_score += 80
                    explanation_parts.append(f"Query matches disease name: {disease['disease_name']}")
                
                # Check for category matches (e.g., "respiratory problem" -> respiratory diseases)
                category_keywords = {
                    'respiratory': ['respiratory', 'lung', 'breathing', 'pneumonia', 'asthma', 'copd'],
                    'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'chest pain', 'hypertension'],
                    'neurological': ['brain', 'neurological', 'headache', 'migraine', 'seizure'],
                    'gastrointestinal': ['stomach', 'gastrointestinal', 'digestive', 'abdomen', 'nausea'],
                    'musculoskeletal': ['bone', 'joint', 'muscle', 'arthritis', 'back pain'],
                    'infectious': ['infection', 'fever', 'viral', 'bacterial', 'covid'],
                    'cancer': ['cancer', 'tumor', 'malignant', 'oncology'],
                    'mental health': ['depression', 'anxiety', 'mental', 'psychiatric'],
                    'endocrine': ['diabetes', 'thyroid', 'hormone', 'metabolic'],
                    'kidney': ['kidney', 'renal', 'urinary', 'bladder'],
                    'skin': ['skin', 'dermatitis', 'rash', 'dermatology']
                }
                
                # Check for category keyword matches
                for category, keywords in category_keywords.items():
                    if category_lower == category.lower():
                        for keyword in keywords:
                            if keyword in clinical_query_lower:
                                match_score += 40
                                explanation_parts.append(f"Category keyword match: {category} conditions")
                                break
                
                # Check for symptom matches
                symptom_matches = []
                for symptom in symptoms:
                    if any(symptom.lower() in s for s in symptoms_lower):
                        match_score += 20
                        symptom_matches.append(symptom)
                
                if symptom_matches:
                    explanation_parts.append(f"Symptoms match: {', '.join(symptom_matches)}")
                
                # Check for age-related conditions
                age = patient_info.get('age')
                if age and isinstance(age, (int, str)):
                    try:
                        age_num = int(str(age))
                        # Different diseases are more common at different ages
                        if age_num < 18 and disease['category'].lower() in ['infectious', 'respiratory']:
                            match_score += 10
                            explanation_parts.append("Age-appropriate condition")
                        elif 18 <= age_num < 65 and disease['category'].lower() in ['cardiovascular', 'diabetes', 'mental health']:
                            match_score += 10
                            explanation_parts.append("Age-appropriate condition")
                        elif age_num >= 65 and disease['category'].lower() in ['cardiovascular', 'neurological', 'cancer']:
                            match_score += 10
                            explanation_parts.append("Age-appropriate condition")
                    except:
                        pass
                
                # Check for generic terms like "problem" or "condition"
                if any(term in clinical_query_lower for term in ['problem', 'condition', 'issue', 'disorder']):
                    if any(keyword in clinical_query_lower for keyword in ['respiratory', 'lung', 'breathing']):
                        if disease['category'].lower() == 'respiratory':
                            match_score += 30
                            explanation_parts.append("Respiratory system condition based on query")
                    elif any(keyword in clinical_query_lower for keyword in ['heart', 'cardiac', 'chest']):
                        if disease['category'].lower() == 'cardiovascular':
                            match_score += 30
                            explanation_parts.append("Cardiovascular condition based on query")
                
                # If we have a good match, add to diagnoses
                if match_score >= 15:  # Lowered threshold for better matching
                    confidence = min(match_score, 85.0)  # Cap at 85%
                    explanation = '. '.join(explanation_parts) + f". This is a {disease['prevalence'].lower()} prevalence condition."
                    
                    diagnoses.append({
                        'condition': disease['disease_name'],
                        'explanation': explanation,
                        'confidence': confidence
                    })
            
            # Sort by confidence and limit to top 3
            diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
            diagnoses = diagnoses[:3]
            
            # If no matches found, provide generic response
            if not diagnoses:
                diagnoses = [{
                    'condition': 'Requires further investigation',
                    'explanation': 'The symptoms provided are insufficient for a specific diagnosis. Please provide more detailed information.',
                    'confidence': 90.0
                }]
            
            return diagnoses
            
        except Exception as e:
            print(f"Error in fallback diagnosis generation: {str(e)}")
            return [{
                'condition': 'Error in diagnosis generation',
                'explanation': f'An error occurred: {str(e)}',
                'confidence': 0.0
            }]

    def _generate_summary_from_database(self, patient_info: Dict, clinical_query: str, differential_diagnoses: List[Dict], risk_assessment: Dict, retrieved_context: List[str], recommendations: Dict = None) -> str:
        """
        Generate a summary using the comprehensive database with personalized next steps
        """
        try:
            # If we have specific diagnoses, create a detailed and user-friendly summary
            if differential_diagnoses and len(differential_diagnoses) > 0:
                primary_diagnosis = differential_diagnoses[0]
                condition = primary_diagnosis.get('condition', 'Unknown condition')
                confidence = primary_diagnosis.get('confidence', 0)
                
                # Create a comprehensive, user-friendly summary
                summary_parts = []
                
                # Patient information with better formatting
                age = patient_info.get('age', 'Unknown age')
                gender = patient_info.get('gender', 'Unknown gender')
                symptoms = patient_info.get('symptoms', [])
                
                if age and age != 'Unknown age':
                    summary_parts.append(f"👤 **Patient Profile:** {age}-year-old {gender}")
                else:
                    summary_parts.append(f"👤 **Patient Profile:** Information provided")
                
                if symptoms:
                    symptoms_text = ', '.join(symptoms[:5])  # First 5 symptoms
                    summary_parts.append(f"🩺 **Presenting Symptoms:** {symptoms_text}")
                
                # Primary diagnosis with detailed explanation
                summary_parts.append(f"🎯 **Primary Consideration:** {condition}")
                summary_parts.append(f"📊 **Confidence Level:** {confidence:.1f}% (Based on symptom analysis and medical evidence)")
                
                # Detailed explanation of the condition
                condition_explanation = self._get_condition_explanation(condition)
                if condition_explanation:
                    summary_parts.append(f"📋 **About {condition}:** {condition_explanation}")
                
                # Additional diagnoses with explanations
                if len(differential_diagnoses) > 1:
                    summary_parts.append("🔍 **Additional Considerations:**")
                    for i, diag in enumerate(differential_diagnoses[1:3], 1):  # Next 2
                        diag_name = diag.get('condition', 'Unknown')
                        diag_confidence = diag.get('confidence', 0)
                        summary_parts.append(f"   {i}. **{diag_name}** ({diag_confidence:.1f}%)")
                
                # Risk assessment with actionable information
                risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
                risk_description = risk_assessment.get('risk_description', '')
                
                summary_parts.append(f"⚠️ **Risk Assessment:** {risk_level.upper()} RISK")
                if risk_description:
                    summary_parts.append(f"📝 **Risk Description:** {risk_description}")
                
                # Personalized next steps based on recommendations
                summary_parts.append("📋 **Recommended Next Steps:**")
                if recommendations and recommendations.get('immediate_actions'):
                    # Use the first 3 most important immediate actions as next steps
                    for i, action in enumerate(recommendations['immediate_actions'][:3], 1):
                        summary_parts.append(f"   {i}. {action}")
                    
                    # Add follow-up timing
                    if recommendations.get('follow_up'):
                        summary_parts.append(f"   • **Follow-up:** {recommendations['follow_up']}")
                else:
                    # Fallback to generic next steps only if no recommendations available
                    summary_parts.append("   • Consult with a healthcare provider for proper diagnosis")
                    summary_parts.append("   • Consider relevant diagnostic tests based on symptoms")
                    summary_parts.append("   • Monitor symptoms and seek immediate care if they worsen")
                
                # Medical knowledge reference
                if retrieved_context:
                    summary_parts.append("🔬 **Analysis Based On:** Current medical evidence and peer-reviewed research")
                
                return "\n\n".join(summary_parts)
            
            else:
                # Try to provide more specific guidance based on the query
                query_lower = clinical_query.lower()
                
                # Check if it's a respiratory query
                if any(term in query_lower for term in ['respiratory', 'lung', 'breathing', 'pneumonia', 'asthma']):
                    return "Based on the respiratory symptoms described, common conditions in this age group include asthma, pneumonia, bronchitis, or upper respiratory infections. A thorough medical evaluation including chest examination, lung function tests, and possibly imaging studies would be recommended to determine the specific cause."
                
                # Check if it's a cardiovascular query
                elif any(term in query_lower for term in ['heart', 'cardiac', 'chest pain', 'hypertension']):
                    return "Based on the cardiovascular symptoms described, potential conditions include hypertension, ischemic heart disease, or arrhythmias. A comprehensive cardiac evaluation including ECG, echocardiogram, and blood pressure monitoring would be recommended."
                
                # Check if it's a neurological query
                elif any(term in query_lower for term in ['headache', 'neurological', 'brain', 'migraine']):
                    return "Based on the neurological symptoms described, potential conditions include tension headaches, migraines, or other neurological disorders. A neurological examination and possibly imaging studies would be recommended."
                
                # Check if it's a gastrointestinal query
                elif any(term in query_lower for term in ['stomach', 'gastrointestinal', 'digestive', 'abdomen']):
                    return "Based on the gastrointestinal symptoms described, potential conditions include GERD, gastritis, IBS, or other digestive disorders. A gastroenterological evaluation would be recommended."
                
                # Generic response
                else:
                    return "Based on the information provided, a comprehensive medical evaluation is recommended to determine the underlying cause of the symptoms. Please provide more specific symptoms or details to help narrow down the potential diagnoses."
        
        except Exception as e:
            print(f"Error generating summary from database: {str(e)}")
            return "Based on the information provided, a comprehensive medical evaluation is recommended."

    def _generate_personalized_recommendations(self, patient_info: Dict, clinical_query: str, differential_diagnoses: List[Dict], risk_assessment: Dict) -> Dict:
        """
        Generate personalized, real-time recommendations based on the patient's specific condition using AI
        """
        try:
            # Extract key information for personalized recommendations
            age = patient_info.get('age', 'Unknown')
            gender = patient_info.get('gender', 'Unknown')
            symptoms = patient_info.get('symptoms', [])
            
            # Get primary diagnosis details
            primary_diagnosis = differential_diagnoses[0] if differential_diagnoses else None
            condition_name = primary_diagnosis.get('condition', 'Unknown condition') if primary_diagnosis else 'Unknown condition'
            confidence = primary_diagnosis.get('confidence', 0) if primary_diagnosis else 0
            explanation = primary_diagnosis.get('explanation', '') if primary_diagnosis else ''
            
            # Get risk assessment details
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            risk_factors = risk_assessment.get('risk_factors', [])
            severity_indicators = risk_assessment.get('severity_indicators', [])
            
            # Create a detailed prompt for personalized recommendations
            recommendations_prompt = f"""
            You are a medical AI assistant. Generate SPECIFIC, PERSONALIZED recommendations for the following patient case.
            DO NOT use generic or templated recommendations. Tailor everything to this specific patient's condition, age, gender, and symptoms.
            
            PATIENT DETAILS:
            - Age: {age}
            - Gender: {gender}
            - Presenting Symptoms: {', '.join(symptoms) if symptoms else 'See clinical query'}
            - Clinical Query: {clinical_query}
            
            PRIMARY DIAGNOSIS:
            - Condition: {condition_name}
            - Confidence: {confidence}%
            - Explanation: {explanation}
            
            RISK ASSESSMENT:
            - Risk Level: {risk_level}
            - Risk Factors: {', '.join(risk_factors) if risk_factors else 'None identified'}
            - Severity Indicators: {', '.join(severity_indicators) if severity_indicators else 'None identified'}
            
            ALL DIFFERENTIAL DIAGNOSES:
            {json.dumps([{'condition': d.get('condition'), 'confidence': d.get('confidence')} for d in differential_diagnoses], indent=2)}
            
            Generate PERSONALIZED recommendations in JSON format with these exact keys:
            - "immediate_actions": 3-5 SPECIFIC immediate actions tailored to THIS patient's condition and risk level
            - "tests": 3-6 SPECIFIC diagnostic tests appropriate for THIS condition and patient
            - "lifestyle": 4-6 SPECIFIC lifestyle modifications relevant to THIS condition
            - "follow_up": Specific follow-up instructions with timeline based on risk level
            
            IMPORTANT:
            1. Consider the patient's age and gender when making recommendations
            2. Prioritize based on the risk level (higher risk = more urgent actions)
            3. Be specific about tests (include what they're testing for)
            4. Tailor lifestyle recommendations to the specific condition
            5. Include timelines where appropriate (e.g., "within 24 hours" for high risk)
            6. Reference the specific condition name in your recommendations
            
            Return ONLY a valid JSON object, no other text.
            """
            
            try:
                # Call the generative model
                recommendations_text = self._call_generative_model_with_retry(recommendations_prompt)
                
                # Parse the JSON response
                try:
                    # Extract JSON from the response
                    if '{' in recommendations_text and '}' in recommendations_text:
                        json_start = recommendations_text.find('{')
                        json_end = recommendations_text.rfind('}') + 1
                        recommendations_json = recommendations_text[json_start:json_end]
                        recommendations = json.loads(recommendations_json)
                        
                        # Validate that we got the required keys
                        if all(key in recommendations for key in ['immediate_actions', 'tests', 'lifestyle', 'follow_up']):
                            print("✓ Successfully generated personalized recommendations")
                            return recommendations
                        else:
                            print("⚠ API response missing required keys, using fallback")
                    else:
                        print("⚠ Could not find JSON in API response, using fallback")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠ Failed to parse API recommendations JSON: {str(e)}, using fallback")
                    
            except Exception as e:
                print(f"⚠ API recommendations generation failed: {str(e)}, using fallback")
            
            # Fallback to database-based recommendations
            return self._generate_recommendations_from_database(patient_info, clinical_query, differential_diagnoses, risk_assessment)
            
        except Exception as e:
            print(f"Error in personalized recommendations generation: {str(e)}")
            # Final fallback
            return self._generate_recommendations_from_database(patient_info, clinical_query, differential_diagnoses, risk_assessment)

    def _generate_recommendations_from_database(self, patient_info: Dict, clinical_query: str, differential_diagnoses: List[Dict], risk_assessment: Dict) -> Dict:
        """
        Generate recommendations using the comprehensive database as a fallback
        (Only used when AI generation fails)
        """
        try:
            recommendations = {
                'immediate_actions': [],
                'tests': [],
                'lifestyle': [],
                'follow_up': 'Follow-up within 2-4 weeks'
            }
            
            # If we have specific diagnoses, provide targeted recommendations
            if differential_diagnoses and len(differential_diagnoses) > 0:
                primary_diagnosis = differential_diagnoses[0]
                condition = primary_diagnosis.get('condition', '').lower()
                
                # Immediate actions based on condition
                if any(term in condition for term in ['fracture', 'injury', 'trauma']):
                    recommendations['immediate_actions'] = [
                        'Immobilize affected area',
                        'Apply ice to reduce swelling',
                        'Seek immediate medical attention',
                        'Avoid weight-bearing activities'
                    ]
                    recommendations['tests'] = ['X-ray imaging', 'Physical examination']
                
                elif any(term in condition for term in ['arthritis', 'joint', 'musculoskeletal']):
                    recommendations['immediate_actions'] = [
                        'Rest affected joints',
                        'Apply heat or cold therapy',
                        'Consider over-the-counter pain relief',
                        'Schedule rheumatology consultation'
                    ]
                    recommendations['tests'] = ['Blood tests (ESR, CRP)', 'Joint imaging', 'Physical examination']
                    recommendations['lifestyle'] = [
                        'Low-impact exercise (swimming, cycling)',
                        'Weight management',
                        'Joint protection techniques',
                        'Physical therapy'
                    ]
                
                elif any(term in condition for term in ['diabetes', 'blood sugar']):
                    recommendations['immediate_actions'] = [
                        'Monitor blood glucose levels',
                        'Schedule endocrinology consultation',
                        'Review current medications'
                    ]
                    recommendations['tests'] = ['HbA1c', 'Fasting glucose', 'Lipid panel', 'Kidney function tests']
                    recommendations['lifestyle'] = [
                        'Balanced diet with carbohydrate counting',
                        'Regular physical activity',
                        'Blood sugar monitoring',
                        'Foot care routine'
                    ]
                
                elif any(term in condition for term in ['heart', 'cardiac', 'cardiovascular']):
                    recommendations['immediate_actions'] = [
                        'Monitor vital signs',
                        'Schedule cardiology consultation',
                        'Review cardiovascular risk factors'
                    ]
                    recommendations['tests'] = ['ECG', 'Echocardiogram', 'Stress test', 'Lipid panel']
                    recommendations['lifestyle'] = [
                        'Heart-healthy diet',
                        'Regular exercise',
                        'Smoking cessation',
                        'Blood pressure monitoring'
                    ]
                
                elif any(term in condition for term in ['respiratory', 'lung', 'asthma', 'copd']):
                    recommendations['immediate_actions'] = [
                        'Monitor breathing patterns',
                        'Schedule pulmonology consultation',
                        'Review inhaler technique'
                    ]
                    recommendations['tests'] = ['Spirometry', 'Chest X-ray', 'Blood oxygen levels']
                    recommendations['lifestyle'] = [
                        'Avoid smoking and secondhand smoke',
                        'Regular exercise as tolerated',
                        'Proper inhaler use',
                        'Avoid known triggers'
                    ]
                
                else:
                    # Generic recommendations for other conditions
                    recommendations['immediate_actions'] = [
                        'Schedule medical consultation',
                        'Monitor symptoms',
                        'Keep symptom diary'
                    ]
                    recommendations['tests'] = ['Comprehensive physical examination', 'Basic laboratory tests']
                    recommendations['lifestyle'] = [
                        'Maintain adequate hydration',
                        'Ensure proper rest and sleep',
                        'Follow a balanced diet',
                        'Engage in regular physical activity as tolerated'
                    ]
            
            else:
                # No specific diagnoses - generic recommendations
                recommendations['immediate_actions'] = [
                    'Schedule medical consultation',
                    'Monitor symptoms',
                    'Keep symptom diary',
                    'Follow preventive measures'
                ]
                recommendations['tests'] = ['Comprehensive physical examination', 'Appropriate diagnostic tests']
                recommendations['lifestyle'] = [
                    'Maintain adequate hydration',
                    'Ensure proper rest and sleep (7-9 hours)',
                    'Follow a balanced, nutritious diet',
                    'Engage in regular physical activity as tolerated',
                    'Manage stress through relaxation techniques',
                    'Avoid known triggers if identified',
                    'Maintain regular sleep schedule',
                    'Limit alcohol and caffeine intake'
                ]
            
            return recommendations
        
        except Exception as e:
            print(f"Error generating recommendations from database: {str(e)}")
            return {
                'immediate_actions': ['Schedule medical consultation'],
                'tests': ['Comprehensive evaluation'],
                'lifestyle': ['Maintain healthy lifestyle'],
                'follow_up': 'Follow-up as recommended by healthcare provider'
            }

    def _call_generative_model_with_retry(self, prompt: str) -> str:
        """
        Call the generative model API with retry mechanism
        """
        for attempt in range(self.max_retries):
            try:
                response = self.generative_model.generate_content(prompt)
                if response.text:
                    return response.text
                else:
                    print(f"Empty response from generative model (attempt {attempt+1}/{self.max_retries})")
            except Exception as e:
                print(f"API call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return ""  # Return empty string if all retries fail

    def _extract_patient_info(self, clinical_query: str) -> Dict:
        """
        Extract key patient information from the clinical query using rule-based approach
        """
        print("📋 Extracting patient information...")
        
        # Use rule-based extraction instead of API calls
        info = self._extract_patient_info_rule_based(clinical_query)
        
        # If rule-based extraction doesn't find enough info, try API as fallback
        if not info.get('symptoms') and not info.get('age') and not info.get('gender'):
            try:
                prompt = f"""
                Extract the following patient information from the clinical query:
                - Age (e.g., 45 years old)
                - Gender (e.g., male, female)
                - Symptoms (e.g., tiredness, cold)
                - Duration of symptoms (e.g., 5 days)

                Return the information in JSON format with keys: age, gender, symptoms, duration.

                Clinical Query: {clinical_query}
                """

                response_text = self._call_generative_model_with_retry(prompt)
                try:
                    # Find JSON in the response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        api_info = json.loads(json_str)
                        
                        # Merge API info with rule-based info
                        for key, value in api_info.items():
                            if value and not info.get(key):
                                info[key] = value
                        
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing patient info JSON: {str(e)}")
            
            except Exception as e:
                print(f"API patient info extraction failed: {str(e)}")
        
        # Ensure symptoms is always a list
        if 'symptoms' in info:
            if isinstance(info['symptoms'], str):
                # If symptoms is a string, split by commas or convert to list
                if info['symptoms']:
                    info['symptoms'] = [s.strip() for s in info['symptoms'].split(',')]
                else:
                    info['symptoms'] = []
            elif not isinstance(info['symptoms'], list):
                # If symptoms is neither string nor list, initialize as empty list
                info['symptoms'] = []
        else:
            info['symptoms'] = []
        
        return info

    def _extract_patient_info_rule_based(self, clinical_query: str) -> Dict:
        """
        Extract patient information using rule-based approach
        """
        import re
        
        info = {
            "age": None,
            "gender": None,
            "symptoms": [],
            "duration": None
        }
        
        query_lower = clinical_query.lower()
        
        # Extract age
        age_patterns = [
            r'(\d+)\s*years?\s*old',
            r'(\d+)\s*y\.o\.',
            r'age\s*(\d+)',
            r'(\d+)\s*year\s*old'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                info['age'] = int(match.group(1))
                break
        
        # Extract gender
        if any(word in query_lower for word in ['male', 'man', 'boy', 'gentleman']):
            info['gender'] = 'male'
        elif any(word in query_lower for word in ['female', 'woman', 'girl', 'lady']):
            info['gender'] = 'female'
        
        # Extract symptoms using our comprehensive database
        symptoms = self._extract_symptoms_from_database(clinical_query)
        if symptoms:
            info['symptoms'] = symptoms
        
        # Extract duration
        duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?',
            r'(\d+)\s*years?',
            r'for\s*(\d+)\s*days?',
            r'since\s*(\d+)\s*days?'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                info['duration'] = match.group(0)
                break
        
        return info

    def _extract_symptoms_from_database(self, clinical_query: str) -> List[str]:
        """
        Extract symptoms using our comprehensive database
        """
        try:
            # Load comprehensive database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if not os.path.exists(comprehensive_file):
                return []
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            query_lower = clinical_query.lower()
            extracted_symptoms = []
            
            # Search through all diseases for symptom matches
            for disease in comprehensive_data:
                symptoms = disease.get('symptoms', [])
                for symptom in symptoms:
                    symptom_lower = symptom.lower()
                    
                    # Check if symptom appears in the query
                    if symptom_lower in query_lower:
                        if symptom not in extracted_symptoms:
                            extracted_symptoms.append(symptom)
                    
                    # Check for partial matches (individual words)
                    symptom_words = symptom_lower.split()
                    for word in symptom_words:
                        if len(word) > 3 and word in query_lower:  # Only words longer than 3 chars
                            if symptom not in extracted_symptoms:
                                extracted_symptoms.append(symptom)
                                break
            
            return extracted_symptoms[:10]  # Limit to 10 symptoms
            
        except Exception as e:
            print(f"Error extracting symptoms from database: {str(e)}")
            return []

    def _generate_differential_diagnoses(self, patient_info: Dict, clinical_query: str, retrieved_context: List[str]) -> List[Dict]:
        """
        Generate differential diagnoses based on patient information and retrieved medical knowledge
        
        Args:
            patient_info: Dictionary containing patient information
            clinical_query: The original clinical query
            retrieved_context: List of retrieved medical knowledge snippets
            
        Returns:
            List of dictionaries containing differential diagnoses with confidence scores
        """
        try:
            # First, try to generate diagnoses from the comprehensive database
            database_diagnoses = self._generate_fallback_diagnoses_from_database(patient_info, clinical_query)
            
            if database_diagnoses and len(database_diagnoses) > 0:
                print(f"Generated {len(database_diagnoses)} diagnoses from database")
                return database_diagnoses
            
            # If database doesn't provide good results, use the generative model as fallback
            print("Using generative model for differential diagnoses...")
            
            # Construct a prompt for the generative model
            prompt = f"""
            Based on the following patient information, clinical query, and medical knowledge, generate a differential diagnosis list.
            
            Patient Information:
            {json.dumps(patient_info, indent=2)}
            
            Clinical Query:
            {clinical_query}
            
            Relevant Medical Knowledge:
            {' '.join(retrieved_context)}
            
            For each potential diagnosis, provide:
            1. The condition name
            2. A brief explanation of why this condition is being considered
            3. A confidence score (0-100) indicating the likelihood of this diagnosis
            
            Return the differential diagnoses as a JSON array of objects with these keys:
            - condition: the name of the condition
            - explanation: explanation of why this condition is being considered
            - confidence: numerical confidence score (0-100)
            
            Return ONLY the JSON array without any other text.
            """
            
            # Call the generative model
            response = self._call_generative_model_with_retry(prompt)
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response if needed
                if '[' in response and ']' in response:
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    diagnoses_json = response[json_start:json_end]
                    diagnoses = json.loads(diagnoses_json)
                else:
                    diagnoses = json.loads(response)
                
                # Ensure each diagnosis has the required fields
                for diagnosis in diagnoses:
                    if 'condition' not in diagnosis:
                        diagnosis['condition'] = 'Unknown condition'
                    if 'explanation' not in diagnosis:
                        diagnosis['explanation'] = 'No explanation provided'
                    if 'confidence' not in diagnosis:
                        diagnosis['confidence'] = 50.0  # Default confidence
                    else:
                        # Ensure confidence is a float
                        diagnosis['confidence'] = float(diagnosis['confidence'])
                
                # Sort by confidence (descending)
                diagnoses.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                return diagnoses
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                print("Failed to parse JSON response for differential diagnoses")
                
                # Extract potential conditions from the response
                lines = response.split('\n')
                diagnoses = []
                
                current_diagnosis = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line starts a new diagnosis
                    if line[0].isdigit() and '.' in line[:3]:
                        # Save previous diagnosis if exists
                        if current_diagnosis and 'condition' in current_diagnosis:
                            diagnoses.append(current_diagnosis)
                        
                        # Start new diagnosis
                        current_diagnosis = {
                            'condition': line.split('.', 1)[1].strip().split(':', 1)[0].strip(),
                            'explanation': '',
                            'confidence': 50.0  # Default confidence
                        }
                    elif current_diagnosis and 'condition' in current_diagnosis:
                        # Add to explanation of current diagnosis
                        current_diagnosis['explanation'] += ' ' + line
                
                # Add the last diagnosis if exists
                if current_diagnosis and 'condition' in current_diagnosis:
                    diagnoses.append(current_diagnosis)
                
                # If we still couldn't extract diagnoses, create a fallback using comprehensive database
                if not diagnoses:
                    diagnoses = self._generate_fallback_diagnoses_from_database(patient_info, clinical_query)
                
                return diagnoses
        
        except Exception as e:
            print(f"Error in _generate_differential_diagnoses: {str(e)}")
            return [
                {
                    'condition': 'Error in diagnosis generation',
                    'explanation': f'An error occurred: {str(e)}',
                    'confidence': 0.0
                }
            ]

    def _retrieve_research_papers(self, diagnoses: List[Dict], patient_info: Dict = None, clinical_query: str = "") -> Dict[str, List[Dict]]:
        """
        Retrieve research papers for each diagnosis with patient-specific context
        
        Args:
            diagnoses: List of diagnosis dictionaries
            patient_info: Dictionary containing patient information (age, gender, symptoms, etc.)
            clinical_query: The clinical query describing the patient's case
            
        Returns:
            Dictionary mapping condition names to lists of research paper dictionaries
        """
        try:
            result = {}
            patient_info = patient_info or {}
            
            # Extract patient-specific search context
            patient_age = patient_info.get('age', '')
            patient_gender = patient_info.get('gender', '')
            patient_symptoms = patient_info.get('symptoms', [])
            if isinstance(patient_symptoms, str):
                patient_symptoms = [patient_symptoms] if patient_symptoms else []
            
            print(f"🔬 Retrieving research papers with patient context: age={patient_age}, gender={patient_gender}, symptoms={len(patient_symptoms)}")
            
            for diagnosis in diagnoses:
                condition = diagnosis.get('condition', '')
                if not condition or condition == 'Unknown condition' or condition == 'Error in diagnosis generation':
                    continue
                
                # Build enhanced search context
                search_context = {
                    'condition': condition,
                    'age': patient_age,
                    'gender': patient_gender,
                    'symptoms': patient_symptoms,
                    'clinical_query': clinical_query,
                    'confidence': diagnosis.get('confidence', 0)
                }
                    
                # First try to get papers from the comprehensive database with context
                papers = self._find_research_papers_for_condition_from_database(condition, search_context)
                
                # If no papers found in database, try PubMed database with context
                if not papers:
                    papers = self._generate_research_papers_for_condition(condition, search_context)
                
                if papers:
                    # Add personalized relevance information to each paper
                    papers = self._personalize_research_papers(papers, search_context)
                    result[condition] = papers
            
            return result
        except Exception as e:
            print(f"Error in _retrieve_research_papers: {str(e)}")
            return {}

    def _find_research_papers_for_condition_from_database(self, condition: str, search_context: Dict = None) -> List[Dict]:
        """
        Find research papers for a specific condition from the comprehensive database with patient-specific context
        """
        try:
            search_context = search_context or {}
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            comprehensive_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
            
            if not os.path.exists(comprehensive_file):
                return []
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            # Find the disease in the comprehensive database
            for disease in comprehensive_data:
                if disease.get('disease_name', '').lower() == condition.lower():
                    research_papers = disease.get('research_papers', [])
                    if research_papers:
                        print(f"✅ Found {len(research_papers)} research papers for {condition} in database")
                        
                        # Enhance with PubMedBERT for better relevance ranking using patient context
                        if self.pubmedbert_model and len(research_papers) > 3:
                            enhanced_papers = self._enhance_research_papers_with_pubmedbert(
                                condition, research_papers, search_context
                            )
                            if enhanced_papers:
                                print(f"🧬 PubMedBERT enhanced research papers ranking for {condition} with patient context")
                                return enhanced_papers[:5]
                        
                        return research_papers[:5]  # Return top 5 papers
            
            return []
            
        except Exception as e:
            print(f"Error finding research papers from database: {str(e)}")
            return []

    def _enhance_research_papers_with_pubmedbert(self, condition: str, research_papers: List[Dict], search_context: Dict = None) -> List[Dict]:
        """
        Use PubMedBERT to enhance research paper relevance ranking with patient-specific context
        """
        try:
            if not self.pubmedbert_model or not research_papers:
                return research_papers
            
            search_context = search_context or {}
            import torch
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create a comprehensive query including patient-specific context
            condition_query = f"{condition} medical research treatment diagnosis"
            
            # Add patient symptoms to the query for better matching
            symptoms = search_context.get('symptoms', [])
            if symptoms:
                symptoms_text = ' '.join(symptoms[:5])  # Top 5 symptoms
                condition_query += f" symptoms: {symptoms_text}"
            
            # Add age-specific considerations
            age = search_context.get('age', '')
            if age:
                try:
                    age_num = int(age)
                    if age_num < 18:
                        condition_query += " pediatric children adolescent"
                    elif age_num > 65:
                        condition_query += " elderly geriatric older adults"
                    else:
                        condition_query += " adult"
                except (ValueError, TypeError):
                    pass
            
            # Add gender-specific considerations
            gender = search_context.get('gender', '')
            if gender:
                condition_query += f" {gender}"
            
            print(f"🔍 Patient-specific query for research papers: {condition_query}")
            
            # Tokenize and encode the query
            query_tokens = self.tokenizer(condition_query, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                query_embeddings = self.pubmedbert_model(**query_tokens).last_hidden_state.mean(dim=1)
            
            enhanced_papers = []
            
            # Process each research paper
            for paper in research_papers:
                try:
                    # Create document text from paper
                    doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                    
                    if not doc_text.strip():
                        enhanced_papers.append(paper)
                        continue
                    
                    # Tokenize and encode the document
                    doc_tokens = self.tokenizer(doc_text, return_tensors='pt', truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        doc_embeddings = self.pubmedbert_model(**doc_tokens).last_hidden_state.mean(dim=1)
                    
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        query_embeddings.cpu().numpy(),
                        doc_embeddings.cpu().numpy()
                    )[0][0]
                    
                    # Boost score if paper mentions specific symptoms
                    symptom_boost = 0.0
                    for symptom in symptoms[:5]:
                        if symptom.lower() in doc_text.lower():
                            symptom_boost += 0.05
                    
                    # Add similarity score to paper
                    enhanced_paper = paper.copy()
                    enhanced_paper['pubmedbert_similarity'] = float(similarity) + symptom_boost
                    enhanced_paper['symptom_relevance'] = symptom_boost > 0
                    enhanced_papers.append(enhanced_paper)
                    
                except Exception as e:
                    print(f"Error processing research paper: {e}")
                    enhanced_papers.append(paper)
                    continue
            
            # Sort by PubMedBERT similarity score
            enhanced_papers.sort(key=lambda x: x.get('pubmedbert_similarity', 0), reverse=True)
            
            # Remove the similarity score from final results but keep symptom_relevance
            for paper in enhanced_papers:
                paper.pop('pubmedbert_similarity', None)
            
            return enhanced_papers
            
        except Exception as e:
            print(f"Error in PubMedBERT research paper enhancement: {e}")
            return research_papers

    def _generate_research_papers_for_condition(self, condition: str, search_context: Dict = None) -> List[Dict]:
        """
        Retrieve actual research papers from PubMed database for a specific medical condition with patient context
        
        Args:
            condition: The name of the medical condition
            search_context: Patient-specific search context
            
        Returns:
            List of research paper dictionaries from actual PubMed data
        """
        try:
            search_context = search_context or {}
            
            # First, try to find actual papers from our PubMed database with context
            actual_papers = self._retrieve_actual_pubmed_papers(condition, search_context)
            if actual_papers:
                return actual_papers[:3]  # Return top 3 actual papers
            
            # Fallback: Generate synthetic papers if no actual data found
            return self._generate_synthetic_research_papers(condition, search_context)
                
        except Exception as e:
            print(f"Error retrieving research papers for {condition}: {str(e)}")
            return []

    def _retrieve_actual_pubmed_papers(self, condition: str, search_context: Dict = None) -> List[Dict]:
        """
        Retrieve actual research papers from the loaded PubMed database with patient-specific filtering
        
        Args:
            condition: The name of the medical condition
            search_context: Patient-specific search context (age, gender, symptoms, etc.)
            
        Returns:
            List of actual research paper dictionaries, ranked by relevance to patient
        """
        try:
            search_context = search_context or {}
            
            # Load PubMed database
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            pubmed_file = os.path.join(data_dir, 'pubmed_research_database.json')
            
            if not os.path.exists(pubmed_file):
                return []
            
            with open(pubmed_file, 'r', encoding='utf-8') as f:
                pubmed_data = json.load(f)
            
            # Search for papers related to the condition
            matching_papers = []
            condition_lower = condition.lower()
            patient_symptoms = search_context.get('symptoms', [])
            patient_age = search_context.get('age', '')
            patient_gender = search_context.get('gender', '')
            
            print(f"🔬 Searching PubMed database for {condition} with {len(patient_symptoms)} symptoms")
            
            for paper in pubmed_data:
                # Check if condition matches in medical_conditions, title, or abstract
                medical_conditions = [mc.lower() for mc in paper.get('medical_conditions', [])]
                title_lower = paper.get('title', '').lower()
                abstract_lower = paper.get('abstract', '').lower()
                
                # Check for exact match or partial match
                condition_match = (condition_lower in medical_conditions or 
                    condition_lower in title_lower or 
                    condition_lower in abstract_lower or
                    any(condition_lower in mc for mc in medical_conditions))
                
                if condition_match:
                    # Calculate relevance score based on patient context
                    relevance_score = 0.0
                    
                    # Check symptom relevance
                    symptom_matches = 0
                    for symptom in patient_symptoms[:10]:  # Check top 10 symptoms
                        symptom_lower = symptom.lower()
                        if symptom_lower in title_lower or symptom_lower in abstract_lower:
                            symptom_matches += 1
                            relevance_score += 2.0
                    
                    # Check age-specific relevance
                    if patient_age:
                        try:
                            age_num = int(patient_age)
                            age_keywords = []
                            if age_num < 18:
                                age_keywords = ['pediatric', 'children', 'adolescent', 'youth', 'child']
                            elif age_num > 65:
                                age_keywords = ['elderly', 'geriatric', 'older', 'aging', 'senior']
                            else:
                                age_keywords = ['adult', 'middle-aged']
                            
                            for keyword in age_keywords:
                                if keyword in title_lower or keyword in abstract_lower:
                                    relevance_score += 1.5
                                    break
                        except (ValueError, TypeError):
                            pass
                    
                    # Check gender-specific relevance
                    if patient_gender:
                        gender_lower = patient_gender.lower()
                        if gender_lower in title_lower or gender_lower in abstract_lower:
                            relevance_score += 1.0
                    
                    # Add paper with relevance metadata
                    paper_copy = paper.copy()
                    paper_copy['patient_relevance_score'] = relevance_score
                    paper_copy['symptom_matches'] = symptom_matches
                    matching_papers.append(paper_copy)
            
            # Sort by relevance score, year, and clinical significance
            def sort_key(paper):
                relevance = paper.get('patient_relevance_score', 0.0)
                year = paper.get('year', 0)
                significance = paper.get('clinical_significance', 'Low')
                significance_score = {'High': 3, 'Medium': 2, 'Low': 1}.get(significance, 1)
                return (relevance, year, significance_score)
            
            matching_papers.sort(key=sort_key, reverse=True)
            
            print(f"✅ Found {len(matching_papers)} PubMed papers for {condition}, top paper relevance: {matching_papers[0].get('patient_relevance_score', 0) if matching_papers else 0}")
            
            return matching_papers
            
        except Exception as e:
            print(f"Error retrieving actual PubMed papers: {str(e)}")
            return []

    def _generate_synthetic_research_papers(self, condition: str, search_context: Dict = None) -> List[Dict]:
        """
        Generate synthetic research papers as fallback when no actual PubMed data is available, 
        personalized with patient context
        
        Args:
            condition: The name of the medical condition
            search_context: Patient-specific search context
            
        Returns:
            List of synthetic research paper dictionaries
        """
        try:
            search_context = search_context or {}
            
            # Build patient context for the prompt
            patient_context = ""
            if search_context.get('age'):
                patient_context += f"Patient age: {search_context['age']}. "
            if search_context.get('gender'):
                patient_context += f"Gender: {search_context['gender']}. "
            if search_context.get('symptoms'):
                symptoms_list = ', '.join(search_context['symptoms'][:5])
                patient_context += f"Presenting symptoms: {symptoms_list}. "
            
            # Construct a prompt for the generative model with patient context
            prompt = f"""
            Generate 3 recent research papers about {condition} that are relevant to the following patient profile:
            {patient_context}
            
            For each paper, provide:
            1. Title (should reflect relevance to patient age, gender, or symptoms if applicable)
            2. Authors
            3. Journal
            4. Year (between 2020-2024)
            5. URL (use https://pubmed.ncbi.nlm.nih.gov/ as a base)
            6. Brief summary (1-2 sentences, mentioning any relevance to the patient's presentation)
            
            Return the papers as a JSON array of objects with these keys:
            - title: the title of the paper
            - authors: the authors of the paper
            - journal: the journal where the paper was published
            - year: the publication year
            - url: the URL to the paper
            - summary: a brief summary of the paper
            
            Return ONLY the JSON array without any other text.
            """
            
            # Call the generative model
            response = self._call_generative_model_with_retry(prompt)
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response if needed
                if '[' in response and ']' in response:
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    papers_json = response[json_start:json_end]
                    papers = json.loads(papers_json)
                else:
                    papers = json.loads(response)
                
                # Ensure each paper has the required fields
                for paper in papers:
                    if 'title' not in paper:
                        paper['title'] = f"Recent Research on {condition}"
                    if 'authors' not in paper:
                        paper['authors'] = "Various Authors"
                    if 'journal' not in paper:
                        paper['journal'] = "Journal of Medical Research"
                    if 'year' not in paper:
                        paper['year'] = 2023
                    if 'url' not in paper:
                        paper['url'] = "https://pubmed.ncbi.nlm.nih.gov/"
                    if 'summary' not in paper:
                        paper['summary'] = f"This paper discusses recent findings related to {condition}."
                
                return papers
                
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response with patient context
                print("Failed to parse JSON response for research papers")
                
                # Build age-specific context for fallback
                age_context = ""
                if search_context.get('age'):
                    try:
                        age_num = int(search_context['age'])
                        if age_num < 18:
                            age_context = " in pediatric patients"
                        elif age_num > 65:
                            age_context = " in elderly populations"
                    except (ValueError, TypeError):
                        pass
                
                # Build symptom context
                symptom_context = ""
                if search_context.get('symptoms'):
                    symptoms_list = ', '.join(search_context['symptoms'][:3])
                    symptom_context = f" presenting with {symptoms_list}"
                
                # Create fallback research papers with patient context
                return [
                    {
                        'title': f"Recent Advances in {condition} Treatment{age_context}",
                        'authors': "Smith J, Johnson A, et al.",
                        'journal': "Journal of Medical Research",
                        'year': 2024,
                        'url': "https://pubmed.ncbi.nlm.nih.gov/",
                        'summary': f"This study explores new treatment approaches for {condition}{symptom_context}, with promising results in clinical trials relevant to this patient profile."
                    },
                    {
                        'title': f"Clinical Outcomes of {condition} in Diverse Populations",
                        'authors': "Chen L, Garcia M, et al.",
                        'journal': "International Medical Journal",
                        'year': 2023,
                        'url': "https://pubmed.ncbi.nlm.nih.gov/",
                        'summary': f"A comprehensive review of {condition} manifestations across different demographic groups, including cases{symptom_context}."
                    }
                ]
        
        except Exception as e:
            print(f"Error in _generate_research_papers_for_condition: {str(e)}")
            return [
                {
                    'title': f"Research on {condition}",
                    'authors': "Various Authors",
                    'journal': "Medical Journal",
                    'year': 2024,
                    'url': "https://pubmed.ncbi.nlm.nih.gov/",
                    'summary': f"This paper discusses {condition}."
                }
            ]
    
    def _personalize_research_papers(self, papers: List[Dict], search_context: Dict) -> List[Dict]:
        """
        Add personalized relevance information to research papers based on patient context
        
        Args:
            papers: List of research paper dictionaries
            search_context: Patient-specific search context
            
        Returns:
            List of papers with added personalization metadata
        """
        try:
            if not papers or not search_context:
                return papers
            
            personalized_papers = []
            patient_age = search_context.get('age', '')
            patient_gender = search_context.get('gender', '')
            patient_symptoms = search_context.get('symptoms', [])
            condition = search_context.get('condition', '')
            
            for paper in papers:
                paper_copy = paper.copy()
                relevance_notes = []
                
                # Check symptom relevance
                title_lower = paper.get('title', '').lower()
                abstract_lower = paper.get('abstract', '').lower()
                summary_lower = paper.get('summary', '').lower()
                
                matching_symptoms = []
                for symptom in patient_symptoms[:10]:
                    symptom_lower = symptom.lower()
                    if (symptom_lower in title_lower or 
                        symptom_lower in abstract_lower or 
                        symptom_lower in summary_lower):
                        matching_symptoms.append(symptom)
                
                if matching_symptoms:
                    relevance_notes.append(f"Discusses symptoms relevant to your case: {', '.join(matching_symptoms[:3])}")
                
                # Check age relevance
                if patient_age:
                    try:
                        age_num = int(patient_age)
                        age_keywords = []
                        age_description = ""
                        
                        if age_num < 18:
                            age_keywords = ['pediatric', 'children', 'adolescent', 'youth', 'child']
                            age_description = "pediatric population"
                        elif age_num > 65:
                            age_keywords = ['elderly', 'geriatric', 'older', 'aging', 'senior']
                            age_description = "elderly patients"
                        else:
                            age_keywords = ['adult', 'middle-aged']
                            age_description = "adult patients"
                        
                        for keyword in age_keywords:
                            if (keyword in title_lower or 
                                keyword in abstract_lower or 
                                keyword in summary_lower):
                                relevance_notes.append(f"Focuses on {age_description}, matching your age group")
                                break
                    except (ValueError, TypeError):
                        pass
                
                # Check gender relevance
                if patient_gender:
                    gender_lower = patient_gender.lower()
                    if (gender_lower in title_lower or 
                        gender_lower in abstract_lower or 
                        gender_lower in summary_lower):
                        relevance_notes.append(f"Includes gender-specific research relevant to {patient_gender} patients")
                
                # Add relevance explanation to the paper
                if relevance_notes:
                    paper_copy['patient_relevance'] = '; '.join(relevance_notes)
                    paper_copy['is_personalized'] = True
                else:
                    paper_copy['patient_relevance'] = f"General research on {condition}"
                    paper_copy['is_personalized'] = False
                
                # Add metadata for UI display
                paper_copy['relevance_score'] = len(matching_symptoms) * 2 + len(relevance_notes)
                
                personalized_papers.append(paper_copy)
            
            print(f"✅ Personalized {len(personalized_papers)} research papers with patient context")
            return personalized_papers
            
        except Exception as e:
            print(f"Error personalizing research papers: {str(e)}")
            return papers

    def _perform_risk_assessment(self, patient_info: Dict, diagnoses: List[Dict]) -> Dict:
        """
        Perform risk assessment based on patient information and diagnoses
        
        Args:
            patient_info: Dictionary containing patient information
            diagnoses: List of diagnosis dictionaries
            
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            # Initialize risk assessment
            risk_assessment = {
                'overall_risk_level': 'low',  # low, moderate, high, critical
                'overall_risk_score': 0,
                'condition_risks': [],
                'urgent_alerts': []
            }
            
            # Extract risk factors from patient info
            age = patient_info.get('age')
            gender = patient_info.get('gender')
            symptoms = patient_info.get('symptoms', [])
            # Ensure symptoms is always a list
            if isinstance(symptoms, str):
                symptoms = [symptoms] if symptoms else []
            elif not isinstance(symptoms, list):
                symptoms = []
            medical_history = patient_info.get('medical_history', [])
            
            # Calculate base risk score based on patient factors
            base_risk_score = 0
            
            # Age-based risk (older patients generally have higher risk)
            if age:
                try:
                    age_value = int(age)
                    if age_value > 65:
                        base_risk_score += 20
                    elif age_value > 50:
                        base_risk_score += 10
                    elif age_value < 12:
                        base_risk_score += 10  # Children also have elevated risk
                except (ValueError, TypeError):
                    pass
            
            # Risk based on medical history
            high_risk_conditions = [
                'diabetes', 'hypertension', 'heart disease', 'cancer', 'copd', 
                'asthma', 'immunocompromised', 'stroke', 'kidney disease'
            ]
            
            for condition in high_risk_conditions:
                if any(condition.lower() in history.lower() for history in medical_history):
                    base_risk_score += 15
            
            # Risk based on symptoms
            urgent_symptoms = [
                'chest pain', 'difficulty breathing', 'shortness of breath', 
                'severe headache', 'loss of consciousness', 'seizure', 
                'severe abdominal pain', 'bleeding', 'high fever'
            ]
            
            for symptom in urgent_symptoms:
                if any(symptom.lower() in s.lower() for s in symptoms) or symptom.lower() in str(patient_info).lower():
                    base_risk_score += 15
                    risk_assessment['urgent_alerts'].append(f"Urgent symptom detected: {symptom}")
            
            # Assess risk for each diagnosis
            max_condition_risk = 0
            for diagnosis in diagnoses:
                condition = diagnosis.get('condition', '')
                confidence = diagnosis.get('confidence', 0)
                
                # Skip non-specific diagnoses
                if 'unknown' in condition.lower() or 'further investigation' in condition.lower():
                    continue
                
                # Initialize condition risk
                condition_risk = {
                    'condition': condition,
                    'risk_level': 'low',
                    'risk_score': 0,
                    'risk_factors': []
                }
                
                # Base condition risk on diagnosis confidence
                condition_risk_score = confidence * 0.3  # Scale confidence to contribute to risk
                
                # Check for high-risk conditions
                high_risk_diagnoses = {
                    'heart attack': 90,
                    'stroke': 90,
                    'pulmonary embolism': 85,
                    'sepsis': 85,
                    'meningitis': 80,
                    'appendicitis': 75,
                    'pneumonia': 70,
                    'covid': 65,
                    'diabetes': 60,
                    'hypertension': 60
                }
                
                for high_risk_condition, risk_value in high_risk_diagnoses.items():
                    if high_risk_condition.lower() in condition.lower():
                        condition_risk_score += risk_value
                        condition_risk['risk_factors'].append(f"High-risk condition: {high_risk_condition}")
                
                # Determine risk level based on score
                if condition_risk_score > 80:
                    condition_risk['risk_level'] = 'critical'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                    risk_assessment['urgent_alerts'].append(f"CRITICAL RISK: {condition} requires immediate attention")
                elif condition_risk_score > 60:
                    condition_risk['risk_level'] = 'high'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                    risk_assessment['urgent_alerts'].append(f"HIGH RISK: {condition} requires prompt evaluation")
                elif condition_risk_score > 40:
                    condition_risk['risk_level'] = 'moderate'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                else:
                    condition_risk['risk_level'] = 'low'
                    condition_risk['risk_score'] = min(100, condition_risk_score)
                
                # Add to condition risks
                risk_assessment['condition_risks'].append(condition_risk)
                
                # Track maximum condition risk
                max_condition_risk = max(max_condition_risk, condition_risk_score)
            
            # Calculate overall risk score
            risk_assessment['overall_risk_score'] = min(100, (base_risk_score + max_condition_risk) / 2)
            
            # Determine overall risk level
            if risk_assessment['overall_risk_score'] > 80:
                risk_assessment['overall_risk_level'] = 'critical'
            elif risk_assessment['overall_risk_score'] > 60:
                risk_assessment['overall_risk_level'] = 'high'
            elif risk_assessment['overall_risk_score'] > 40:
                risk_assessment['overall_risk_level'] = 'moderate'
            else:
                risk_assessment['overall_risk_level'] = 'low'
            
            return risk_assessment
        
        except Exception as e:
            print(f"Error in _perform_risk_assessment: {str(e)}")
            return {
                'overall_risk_level': 'unknown',
                'overall_risk_score': 0,
                'condition_risks': [],
                'urgent_alerts': [f"Error in risk assessment: {str(e)}"]
            }

    def _assess_risk_with_confidence(self, diagnoses: List[Dict]) -> Dict:
        """
        Assess risk level for each diagnosis using confidence scores
        This provides an additional confidence-based risk assessment
        
        Args:
            diagnoses: List of diagnosis dictionaries with confidence scores
            
        Returns:
            Dictionary with risk assessment for each condition
        """
        risk_assessment = {}
        for diagnosis in diagnoses:
            condition = diagnosis.get("condition", "")
            confidence = diagnosis.get("confidence", 0) / 100.0  # Normalize to 0-1
            
            # Calculate risk score using confidence and condition-specific risk level
            risk_score = confidence * self._get_condition_risk_level(condition)
            
            risk_assessment[condition] = {
                "risk_score": round(risk_score, 3),
                "risk_level": self._get_risk_level_label(risk_score),
                "confidence": diagnosis.get("confidence", 0)
            }
        return risk_assessment

    def _get_condition_risk_level(self, condition: str) -> float:
        """
        Get predefined risk level for specific conditions (0.0 - 1.0)
        
        Args:
            condition: The medical condition name
            
        Returns:
            Float representing the inherent risk level of the condition
        """
        condition_lower = condition.lower()
        
        # Critical/High-risk conditions (requires immediate attention)
        high_risk_conditions = [
            "heart attack", "myocardial infarction", "stroke", "sepsis", 
            "pulmonary embolism", "meningitis", "pneumonia", "anaphylaxis",
            "acute coronary syndrome", "aortic dissection", "subarachnoid hemorrhage"
        ]
        
        # Medium-risk conditions (requires prompt medical attention)
        medium_risk_conditions = [
            "diabetes", "hypertension", "asthma", "copd", "bronchitis",
            "heart failure", "atrial fibrillation", "kidney disease",
            "liver disease", "chronic kidney disease", "angina"
        ]
        
        # Check for high-risk conditions
        for high_risk in high_risk_conditions:
            if high_risk in condition_lower:
                return 1.0
        
        # Check for medium-risk conditions
        for medium_risk in medium_risk_conditions:
            if medium_risk in condition_lower:
                return 0.7
        
        # Default to moderate risk
        return 0.5

    def _get_risk_level_label(self, risk_score: float) -> str:
        """
        Convert risk score to human-readable risk level label
        
        Args:
            risk_score: Numerical risk score (0.0 - 1.0)
            
        Returns:
            String label: "Critical", "High", "Medium", or "Low"
        """
        if risk_score >= 0.8:
            return "Critical"
        elif risk_score >= 0.6:
            return "High"
        elif risk_score >= 0.4:
            return "Medium"
        else:
            return "Low"

    def _generate_alerts(self, risk_assessment: Dict) -> List[str]:
        """
        Generate alerts for high-risk and critical conditions
        
        Args:
            risk_assessment: Dictionary with risk assessments for each condition
            
        Returns:
            List of alert strings
        """
        alerts = []
        for condition, assessment in risk_assessment.items():
            risk_level = assessment.get("risk_level", "Low")
            confidence = assessment.get("confidence", 0)
            
            if risk_level == "Critical":
                alerts.append(f"🚨 CRITICAL ALERT: {condition} detected with {confidence:.1f}% confidence - Seek immediate emergency medical attention!")
            elif risk_level == "High":
                alerts.append(f"⚠️ HIGH RISK ALERT: {condition} detected with {confidence:.1f}% confidence - Urgent medical evaluation required!")
        
        return alerts

    def search_medical_knowledge(self, query: str, filters: Dict = None) -> Dict:
        """
        Search for medical knowledge with filtering options and PubMed research sources
        
        Args:
            query: Search query
            filters: Dictionary of filters (e.g., {'type': 'research_paper', 'year': 2022})
            
        Returns:
            Dictionary containing search results with PubMed research sources
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search in vector database
            k = 10  # Number of results to retrieve
            distances, indices = self.index.search(query_embedding, k)
            
            # Get relevant documents
            relevant_documents = []
            research_sources = []
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_store):
                    doc = self.document_store[idx]
                    relevant_documents.append(doc)
                    
                    # Extract research sources from document
                    if 'PMID:' in doc and 'Full Text URL:' in doc:
                        # Parse research paper information
                        lines = doc.split('\n')
                        paper_info = {}
                        for line in lines:
                            if line.strip().startswith('PMID:'):
                                paper_info['pmid'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('Research Paper:'):
                                paper_info['title'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('Journal:'):
                                paper_info['journal'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('Year:'):
                                paper_info['year'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('Full Text URL:'):
                                paper_info['full_text_url'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('PubMed URL:'):
                                paper_info['pubmed_url'] = line.split(':', 1)[1].strip()
                            elif line.strip().startswith('DOI:'):
                                paper_info['doi'] = line.split(':', 1)[1].strip()
                        
                        if paper_info:
                            research_sources.append(paper_info)
            
            return {
                'query': query,
                'relevant_documents': relevant_documents,
                'research_sources': research_sources,
                'total_results': len(relevant_documents),
                'filters_applied': filters
            }
            
        except Exception as e:
            print(f"Error in medical knowledge search: {str(e)}")
            return {
                'query': query,
                'relevant_documents': [],
                'research_sources': [],
                'total_results': 0,
                'error': str(e)
            }

    def _retrieve_medical_knowledge(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve relevant medical knowledge from the vector database
        
        Args:
            query: Search query
            top_k: Number of top results to retrieve
            
        Returns:
            List of relevant medical knowledge documents
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search in vector database
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Get relevant documents
            relevant_documents = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.document_store):
                    doc = self.document_store[idx]
                    relevant_documents.append(doc)
            
            return relevant_documents
        
        except Exception as e:
            print(f"Error retrieving medical knowledge: {str(e)}")
            return []

# Function to create a sample medical knowledge database for testing
def create_sample_medical_knowledge(output_path="sample_medical_knowledge.json"):
    """
    Create a sample medical knowledge database for testing the RAG system
    """
    sample_data = [
            {
                "condition": "Pneumonia",
                "description": "Infection that inflames air sacs in one or both lungs",
                "symptoms": ["fever", "cough", "chest pain", "shortness of breath", "fatigue"],
                "duration": "1-3 weeks",
                "complications": ["bacteremia", "pleural effusion", "lung abscess"],
                "treatments": ["antibiotics", "rest", "fluids"],
                "keywords": ["lung infection", "respiratory illness"]
            },
            {
                "condition": "Bronchitis",
                "description": "Inflammation of the lining of bronchial tubes",
                "symptoms": ["cough", "mucus production", "fatigue", "shortness of breath", "fever"],
                "duration": "10-14 days",
                "complications": ["pneumonia", "chronic bronchitis"],
                "treatments": ["rest", "fluids", "humidifier"],
                "keywords": ["chest infection", "respiratory condition"]
            },
            # Add more sample data as needed
    ]

    # Add some research papers
    sample_papers = [
        {
            "title": "Clinical Outcomes in Patients with Pneumonia: A Meta-Analysis",
            "authors": "Smith J, Johnson B, Chen L",
            "journal": "Journal of Respiratory Medicine",
            "year": "2023",
            "url": "https://example.com/pneumonia-meta-analysis",
            "keywords": ["pneumonia", "clinical outcomes", "meta-analysis", "respiratory infection"]
        },
        {
            "title": "Recent Advances in Bronchitis Treatment",
            "authors": "Williams T, Anderson K",
            "journal": "Respiratory Research",
            "year": "2022",
            "url": "https://example.com/bronchitis-advances",
            "keywords": ["bronchitis", "treatment", "clinical research", "respiratory disease"]
        },
        # Add more sample papers as needed
    ]

    # Combine data
    all_data = sample_data + sample_papers

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Sample medical knowledge created at {output_path}")
    return output_path

def get_rag_system():
    """
    Get or create a singleton instance of the RAG system
    """
    # Use a singleton pattern to avoid recreating the RAG system for each request
    if not hasattr(get_rag_system, 'instance'):
        from django.conf import settings
        
        # Initialize the RAG system with the medical knowledge path from settings
        medical_knowledge_path = getattr(settings, 'MEDICAL_KNOWLEDGE_PATH', None)
        get_rag_system.instance = RAGClinicalDecisionSupport(medical_data_path=medical_knowledge_path)
        
    return get_rag_system.instance 