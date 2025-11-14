"""
Advanced LLM and Machine Translation Features for Medical CDSS

This module implements:
1. Dense Retrieval using PubMedBERT transformer models
2. Enhanced RAG Generator with advanced prompt engineering
3. Automated Medical Text Summarization (extractive and abstractive)
4. Question Answering capabilities with transformer models
5. Optimized vector embeddings and FAISS retrieval

Uses state-of-the-art transformer models for medical text understanding.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForQuestionAnswering,
    pipeline, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import logging
# Alternative text summarization using simple extractive method
import heapq
from collections import Counter
import google.generativeai as genai
from django.conf import settings

logger = logging.getLogger(__name__)

class DenseRetrieval:
    """Advanced dense retrieval using transformer models for medical text"""
    
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        """
        Initialize dense retrieval with biomedical transformer model
        
        Args:
            model_name: Name of the sentence transformer model
                       Options: 'pritamdeka/S-PubMedBert-MS-MARCO' (biomedical)
                               'all-mpnet-base-v2' (general purpose)
                               'dmis-lab/biobert-base-cased-v1.1' (BioBERT)
        """
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading dense retrieval model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Dense retrieval model loaded (dimension: {self.embedding_dim})")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, falling back to all-mpnet-base-v2")
            self.model = SentenceTransformer('all-mpnet-base-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to dense vector embeddings
        
        Args:
            texts: List of text documents to encode
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        try:
            # Preprocess texts for medical content
            cleaned_texts = [self._preprocess_medical_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess text for better medical understanding"""
        # Expand common medical abbreviations for better embeddings
        abbreviations = {
            r'\bMI\b': 'myocardial infarction',
            r'\bBP\b': 'blood pressure',
            r'\bHR\b': 'heart rate',
            r'\bPE\b': 'pulmonary embolism',
            r'\bCT\b': 'computed tomography',
            r'\bMRI\b': 'magnetic resonance imaging',
            r'\bECG\b': 'electrocardiogram',
            r'\bICU\b': 'intensive care unit'
        }
        
        processed_text = text
        for abbr, expansion in abbreviations.items():
            processed_text = re.sub(abbr, f"{expansion} ({abbr.strip('\\b')})", processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def find_similar(self, query: str, document_embeddings: np.ndarray, 
                    documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        Find most similar documents using dense retrieval
        
        Args:
            query: Search query
            document_embeddings: Pre-computed document embeddings
            documents: Original document texts
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document, score, and index
        """
        try:
            # Encode query
            query_embedding = self.encode_texts([query])[0]
            
            # Calculate cosine similarity
            similarities = np.dot(document_embeddings, query_embedding) / (
                np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'document': documents[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx),
                    'method': 'dense_retrieval'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []

class MedicalTextSummarizer:
    """Advanced medical text summarization with extractive and abstractive methods"""
    
    def __init__(self):
        """Initialize the medical text summarizer"""
        self._load_models()
    
    def _load_models(self):
        """Load summarization models"""
        try:
            # Load summarization pipeline
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=130,
                min_length=30,
                do_sample=False
            )
            print("✅ Abstractive summarization model loaded")
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {e}")
            self.summarization_pipeline = None
    
    def extractive_summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Extractive summarization using simple sentence scoring
        
        Args:
            text: Input text to summarize
            ratio: Ratio of sentences to keep (0.1 = 10% of sentences)
            
        Returns:
            Extractive summary
        """
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) < 3:
                return text  # Too short to summarize
            
            # Simple scoring based on word frequency
            words = text.lower().split()
            word_freq = Counter(words)
            
            # Score sentences based on word frequencies
            sentence_scores = {}
            for sentence in sentences:
                sentence_words = sentence.lower().split()
                score = 0
                word_count = 0
                for word in sentence_words:
                    if word in word_freq:
                        score += word_freq[word]
                        word_count += 1
                if word_count > 0:
                    sentence_scores[sentence] = score / word_count
            
            # Select top sentences
            num_sentences = max(1, int(len(sentences) * ratio))
            top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    summary_sentences.append(sentence)
            
            summary = '. '.join(summary_sentences) + '.'
            return summary
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def abstractive_summarize(self, text: str, max_length: int = 130) -> str:
        """
        Abstractive summarization using transformer models
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Abstractive summary
        """
        try:
            if not self.summarization_pipeline:
                return self.extractive_summarize(text)
            
            # Split long texts into chunks
            max_input_length = 1024
            if len(text) > max_input_length:
                # Summarize in chunks and then combine
                chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]
                chunk_summaries = []
                
                for chunk in chunks[:3]:  # Limit to 3 chunks
                    if len(chunk.strip()) > 50:
                        result = self.summarization_pipeline(chunk, max_length=max_length//len(chunks))
                        chunk_summaries.append(result[0]['summary_text'])
                
                return ' '.join(chunk_summaries)
            
            # Summarize single text
            result = self.summarization_pipeline(text, max_length=max_length)
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            return self.extractive_summarize(text)
    
    def medical_summary(self, text: str, summary_type: str = "abstractive") -> Dict[str, str]:
        """
        Generate medical text summary with both methods
        
        Args:
            text: Medical text to summarize
            summary_type: "extractive", "abstractive", or "both"
            
        Returns:
            Dictionary with summary results
        """
        result = {
            'original_length': len(text.split()),
            'extractive_summary': '',
            'abstractive_summary': '',
            'key_findings': []
        }
        
        if summary_type in ["extractive", "both"]:
            result['extractive_summary'] = self.extractive_summarize(text)
        
        if summary_type in ["abstractive", "both"]:
            result['abstractive_summary'] = self.abstractive_summarize(text)
        
        # Extract key medical findings
        result['key_findings'] = self._extract_key_findings(text)
        
        return result
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key medical findings from text"""
        findings = []
        
        # Medical finding patterns
        finding_patterns = [
            r'diagnosed with ([^.]+)',
            r'shows signs of ([^.]+)',
            r'symptoms include ([^.]+)',
            r'treatment with ([^.]+)',
            r'presents with ([^.]+)',
            r'findings suggest ([^.]+)'
        ]
        
        for pattern in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend([match.strip() for match in matches])
        
        return list(set(findings))  # Remove duplicates

class EnhancedRAGGenerator:
    """Enhanced RAG Generator with advanced prompt engineering for medical contexts"""
    
    def __init__(self):
        """Initialize the enhanced RAG generator"""
        self.api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Medical prompt templates
        self.prompt_templates = {
            'diagnosis': self._get_diagnosis_prompt_template(),
            'treatment': self._get_treatment_prompt_template(),
            'summary': self._get_summary_prompt_template(),
            'general': self._get_general_prompt_template()
        }
    
    def _get_diagnosis_prompt_template(self) -> str:
        """Prompt template for diagnostic queries"""
        return """You are an expert medical assistant specializing in clinical diagnosis. Analyze the provided medical context and patient information to generate accurate diagnostic insights.

**Instructions:**
- Base your analysis STRICTLY on the provided medical context
- Consider differential diagnoses with confidence levels
- Highlight key symptoms and their clinical significance
- Suggest appropriate diagnostic tests if relevant
- Use evidence-based medical knowledge

**Medical Context:**
{context}

**Patient Query:**
{query}

**Response Format:**
Please provide your analysis in the following structured format:

1. **Summary:** Brief overview of the case
2. **Key Findings:** Most significant symptoms/signs
3. **Differential Diagnoses:** List potential diagnoses with confidence levels
4. **Recommended Actions:** Diagnostic tests or immediate interventions
5. **Evidence Sources:** Cite relevant medical literature from the context

**Response:**"""

    def _get_treatment_prompt_template(self) -> str:
        """Prompt template for treatment queries"""
        return """You are an expert medical assistant specializing in treatment planning. Provide evidence-based treatment recommendations based on the medical context provided.

**Instructions:**
- Recommend treatments based ONLY on the provided medical evidence
- Consider contraindications and patient safety
- Suggest both pharmacological and non-pharmacological interventions
- Include monitoring parameters where appropriate

**Medical Context:**
{context}

**Treatment Query:**
{query}

**Response Format:**
1. **Treatment Overview:** Primary treatment approach
2. **Pharmacological Interventions:** Medications with dosages
3. **Non-pharmacological Interventions:** Lifestyle, therapy recommendations
4. **Monitoring:** What to monitor during treatment
5. **Follow-up:** When to reassess
6. **Evidence Sources:** Supporting literature from context

**Response:**"""

    def _get_summary_prompt_template(self) -> str:
        """Prompt template for medical text summarization"""
        return """You are a medical expert tasked with summarizing complex medical information for healthcare professionals.

**Instructions:**
- Create a concise, accurate summary of the medical context
- Preserve all critical medical information
- Use clear, professional medical terminology
- Organize information logically

**Medical Context to Summarize:**
{context}

**Query Focus:**
{query}

**Response Format:**
1. **Executive Summary:** Key points in 2-3 sentences
2. **Clinical Details:** Important medical findings
3. **Recommendations:** Action items or next steps
4. **Sources:** Key references mentioned

**Response:**"""

    def _get_general_prompt_template(self) -> str:
        """General prompt template for medical queries"""
        return """You are a knowledgeable medical assistant helping healthcare professionals with clinical questions.

**Instructions:**
- Answer based STRICTLY on the provided medical context
- If information is insufficient, clearly state limitations
- Use evidence-based medical knowledge
- Provide practical, actionable insights

**Medical Context:**
{context}

**Query:**
{query}

**Response Format:**
1. **Direct Answer:** Address the specific question
2. **Supporting Evidence:** Relevant details from context
3. **Clinical Implications:** Practical significance
4. **Additional Considerations:** Other relevant factors
5. **Sources:** References from the provided context

**Response:**"""

    def generate_response(self, query: str, retrieved_context: List[str], 
                         response_type: str = "general") -> Dict[str, Any]:
        """
        Generate enhanced RAG response with advanced prompt engineering
        
        Args:
            query: User's medical query
            retrieved_context: List of retrieved medical documents
            response_type: Type of response ("diagnosis", "treatment", "summary", "general")
            
        Returns:
            Dictionary with generated response and metadata
        """
        try:
            # Prepare context
            context = self._prepare_context(retrieved_context)
            
            # Select appropriate prompt template
            template = self.prompt_templates.get(response_type, self.prompt_templates['general'])
            
            # Format prompt
            prompt = template.format(context=context, query=query)
            
            # Generate response
            response = self.model.generate_content(prompt)
            response_text = response.text if response else "Unable to generate response"
            
            # Parse structured response
            parsed_response = self._parse_structured_response(response_text)
            
            return {
                'response': response_text,
                'parsed_response': parsed_response,
                'response_type': response_type,
                'context_length': len(context),
                'prompt_length': len(prompt),
                'sources_count': len(retrieved_context)
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'parsed_response': {},
                'response_type': response_type,
                'context_length': 0,
                'prompt_length': 0,
                'sources_count': 0
            }
    
    def _prepare_context(self, retrieved_docs: List[str]) -> str:
        """Prepare and format retrieved context for the prompt"""
        if not retrieved_docs:
            return "No relevant medical context available."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5 documents
            context_parts.append(f"**Source {i}:**\n{doc.strip()}\n")
        
        return "\n".join(context_parts)
    
    def _parse_structured_response(self, response_text: str) -> Dict[str, str]:
        """Parse structured response into components"""
        sections = {}
        
        # Common section patterns
        patterns = {
            'summary': r'\*\*Summary:?\*\*\s*(.*?)(?=\*\*|\n\n|$)',
            'key_findings': r'\*\*Key Findings:?\*\*\s*(.*?)(?=\*\*|\n\n|$)',
            'diagnoses': r'\*\*Differential Diagnoses:?\*\*\s*(.*?)(?=\*\*|\n\n|$)',
            'recommendations': r'\*\*Recommended Actions:?\*\*\s*(.*?)(?=\*\*|\n\n|$)',
            'treatment': r'\*\*Treatment:?\*\*\s*(.*?)(?=\*\*|\n\n|$)',
            'sources': r'\*\*Evidence Sources:?\*\*\s*(.*?)(?=\*\*|\n\n|$)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section] = match.group(1).strip()
        
        return sections

class QuestionAnsweringSystem:
    """Medical Question Answering using BERT-based models"""
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize QA system with BERT model
        
        Args:
            model_name: Name of the QA model
        """
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load the question answering model"""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name
            )
            print(f"✅ Question Answering model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load QA model: {e}")
            self.qa_pipeline = None
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on provided context
        
        Args:
            question: Medical question to answer
            context: Medical context/document
            
        Returns:
            Dictionary with answer and confidence score
        """
        try:
            if not self.qa_pipeline:
                return {
                    'answer': 'Question answering model not available',
                    'score': 0.0,
                    'start': 0,
                    'end': 0
                }
            
            result = self.qa_pipeline(question=question, context=context)
            
            return {
                'answer': result['answer'],
                'score': result['score'],
                'start': result['start'],
                'end': result['end'],
                'confidence': 'high' if result['score'] > 0.8 else 'medium' if result['score'] > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {
                'answer': f'Error processing question: {str(e)}',
                'score': 0.0,
                'start': 0,
                'end': 0,
                'confidence': 'low'
            }

# Global instances for easy access
_dense_retrieval = None
_text_summarizer = None
_rag_generator = None
_qa_system = None

def get_dense_retrieval() -> DenseRetrieval:
    """Get or create global dense retrieval instance"""
    global _dense_retrieval
    if _dense_retrieval is None:
        _dense_retrieval = DenseRetrieval()
    return _dense_retrieval

def get_text_summarizer() -> MedicalTextSummarizer:
    """Get or create global text summarizer instance"""
    global _text_summarizer
    if _text_summarizer is None:
        _text_summarizer = MedicalTextSummarizer()
    return _text_summarizer

def get_rag_generator() -> EnhancedRAGGenerator:
    """Get or create global RAG generator instance"""
    global _rag_generator
    if _rag_generator is None:
        _rag_generator = EnhancedRAGGenerator()
    return _rag_generator

def get_qa_system() -> QuestionAnsweringSystem:
    """Get or create global QA system instance"""
    global _qa_system
    if _qa_system is None:
        _qa_system = QuestionAnsweringSystem()
    return _qa_system

# Convenience functions
def dense_retrieve(query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
    """Convenience function for dense retrieval"""
    retriever = get_dense_retrieval()
    doc_embeddings = retriever.encode_texts(documents)
    return retriever.find_similar(query, doc_embeddings, documents, top_k)

def summarize_medical_text(text: str, method: str = "abstractive") -> Dict[str, str]:
    """Convenience function for medical text summarization"""
    summarizer = get_text_summarizer()
    return summarizer.medical_summary(text, method)

def generate_rag_response(query: str, context: List[str], response_type: str = "general") -> Dict[str, Any]:
    """Convenience function for RAG response generation"""
    generator = get_rag_generator()
    return generator.generate_response(query, context, response_type)

def answer_medical_question(question: str, context: str) -> Dict[str, Any]:
    """Convenience function for medical question answering"""
    qa_system = get_qa_system()
    return qa_system.answer_question(question, context)
