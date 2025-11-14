#!/usr/bin/env python3
"""
Lightweight RAG Clinical Decision Support System
Optimized for memory efficiency and faster responses
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from collections import defaultdict

class LightweightRAGSystem:
    """
    Lightweight RAG system optimized for memory efficiency
    """
    
    def __init__(self, medical_data_path: Optional[str] = None):
        """
        Initialize the lightweight RAG system
        
        Args:
            medical_data_path: Path to medical knowledge data
        """
        print("🚀 Initializing Lightweight RAG Clinical Decision Support System...")
        
        # Initialize basic components
        self.medical_data_path = medical_data_path or "Rag/data"
        self.document_store = {}
        self.symptom_lexicon = {}
        self.medical_knowledge = []
        self.drug_database = []
        self.clinical_guidelines = []
        self.research_papers = []
        self.loaded_documents = 0
        
        # Load medical knowledge
        self._load_medical_knowledge()
        
        print(f"✅ Lightweight RAG system initialized with {self.loaded_documents} documents")
    
    def _load_medical_knowledge(self):
        """Load medical knowledge from JSON files"""
        print("📚 Loading medical knowledge...")
        
        data_path = Path(self.medical_data_path)
        if not data_path.exists():
            print(f"❌ Data path not found: {data_path}")
            return
        
        # Load different types of medical data
        self._load_comprehensive_medical_knowledge(data_path)
        self._load_drug_database(data_path)
        self._load_clinical_guidelines(data_path)
        self._load_research_papers(data_path)
        self._load_symptom_lexicon(data_path)
        
        print(f"✅ Loaded {self.loaded_documents} medical documents")
    
    def _load_comprehensive_medical_knowledge(self, data_path: Path):
        """Load comprehensive medical knowledge"""
        file_path = data_path / "comprehensive_medical_knowledge.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.medical_knowledge = json.load(f)
                print(f"✅ Loaded {len(self.medical_knowledge)} medical conditions")
            except Exception as e:
                print(f"❌ Error loading medical knowledge: {e}")
    
    def _load_drug_database(self, data_path: Path):
        """Load drug database"""
        file_path = data_path / "comprehensive_drug_database.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.drug_database = json.load(f)
                print(f"✅ Loaded {len(self.drug_database)} drugs")
            except Exception as e:
                print(f"❌ Error loading drug database: {e}")
    
    def _load_clinical_guidelines(self, data_path: Path):
        """Load clinical guidelines"""
        file_path = data_path / "clinical_guidelines_database.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.clinical_guidelines = json.load(f)
                print(f"✅ Loaded {len(self.clinical_guidelines)} clinical guidelines")
            except Exception as e:
                print(f"❌ Error loading clinical guidelines: {e}")
    
    def _load_research_papers(self, data_path: Path):
        """Load research papers"""
        research_files = [
            "pubmed_research_database.json",
            "additional_medical_research.json"
        ]
        
        for filename in research_files:
            file_path = data_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                        self.research_papers.extend(papers)
                    print(f"✅ Loaded {len(papers)} research papers from {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
    
    def _load_symptom_lexicon(self, data_path: Path):
        """Load symptom-disease lexicon"""
        file_path = data_path / "expanded_symptom_disease_lexicon.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.symptom_lexicon = json.load(f)
                print(f"✅ Loaded symptom lexicon with {len(self.symptom_lexicon)} conditions")
            except Exception as e:
                print(f"❌ Error loading symptom lexicon: {e}")
    
    def search_medical_knowledge(self, query: str, filters: Dict = None) -> Dict:
        """
        Search medical knowledge using keyword matching
        
        Args:
            query: Search query
            filters: Optional filters
            
        Returns:
            Dictionary with search results
        """
        try:
            query_lower = query.lower()
            results = {
                'query': query,
                'medical_conditions': [],
                'drugs': [],
                'clinical_guidelines': [],
                'research_papers': [],
                'total_results': 0
            }
            
            # Search medical conditions
            for condition in self.medical_knowledge:
                if self._matches_query(condition, query_lower):
                    results['medical_conditions'].append(condition)
            
            # Search drugs
            for drug in self.drug_database:
                if self._matches_query(drug, query_lower):
                    results['drugs'].append(drug)
            
            # Search clinical guidelines
            for guideline in self.clinical_guidelines:
                if self._matches_query(guideline, query_lower):
                    results['clinical_guidelines'].append(guideline)
            
            # Search research papers
            for paper in self.research_papers:
                if self._matches_query(paper, query_lower):
                    results['research_papers'].append(paper)
            
            results['total_results'] = (
                len(results['medical_conditions']) + 
                len(results['drugs']) + 
                len(results['clinical_guidelines']) + 
                len(results['research_papers'])
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error in medical knowledge search: {e}")
            return {
                'query': query,
                'error': str(e),
                'total_results': 0
            }
    
    def _matches_query(self, item: Dict, query: str) -> bool:
        """Check if an item matches the query"""
        # Extract key terms from query
        query_terms = query.lower().split()
        
        # Search in various fields
        searchable_fields = [
            'condition', 'title', 'drug_name', 'generic_name',
            'symptoms', 'keywords', 'description', 'abstract',
            'indications', 'medical_conditions'
        ]
        
        match_count = 0
        total_terms = len(query_terms)
        
        for field in searchable_fields:
            if field in item:
                value = item[field]
                if isinstance(value, str):
                    value_lower = value.lower()
                    for term in query_terms:
                        if term in value_lower:
                            match_count += 1
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str):
                            v_lower = v.lower()
                            for term in query_terms:
                                if term in v_lower:
                                    match_count += 1
        
        # Return True if at least one term matches
        return match_count > 0
    
    def analyze_medical_query(self, query: str) -> Dict:
        """
        Analyze a medical query and provide comprehensive response
        
        Args:
            query: Medical query
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract symptoms from query
            symptoms = self._extract_symptoms(query)
            
            # Search for relevant medical knowledge
            search_results = self.search_medical_knowledge(query)
            
            # Generate differential diagnoses
            differential_diagnoses = self._generate_differential_diagnoses(symptoms, search_results)
            
            # Generate treatment recommendations
            treatment_recommendations = self._generate_treatment_recommendations(search_results)
            
            # Generate research sources
            research_sources = self._extract_research_sources(search_results)
            
            return {
                'query': query,
                'extracted_symptoms': symptoms,
                'differential_diagnoses': differential_diagnoses,
                'treatment_recommendations': treatment_recommendations,
                'research_sources': research_sources,
                'search_results': search_results,
                'confidence_score': self._calculate_confidence_score(symptoms, search_results)
            }
            
        except Exception as e:
            print(f"❌ Error in medical query analysis: {e}")
            return {
                'query': query,
                'error': str(e),
                'differential_diagnoses': [],
                'treatment_recommendations': [],
                'research_sources': []
            }
    
    def _extract_symptoms(self, query: str) -> List[str]:
        """Extract symptoms from query using keyword matching"""
        symptoms = []
        query_lower = query.lower()
        
        # Check against symptom lexicon
        for condition, info in self.symptom_lexicon.items():
            condition_symptoms = info.get('symptoms', [])
            for symptom in condition_symptoms:
                if symptom.lower() in query_lower:
                    symptoms.append(symptom)
        
        return list(set(symptoms))  # Remove duplicates
    
    def _generate_differential_diagnoses(self, symptoms: List[str], search_results: Dict) -> List[Dict]:
        """Generate differential diagnoses based on symptoms and search results"""
        diagnoses = []
        
        # Score conditions based on symptom matches
        condition_scores = {}
        
        for condition in self.medical_knowledge:
            condition_symptoms = condition.get('symptoms', [])
            score = 0
            
            for symptom in symptoms:
                for cond_symptom in condition_symptoms:
                    if symptom.lower() in cond_symptom.lower():
                        score += 1
            
            if score > 0:
                condition_scores[condition['condition']] = {
                    'condition': condition['condition'],
                    'score': score,
                    'confidence': min(score / len(symptoms), 1.0),
                    'symptoms_matched': score,
                    'total_symptoms': len(condition_symptoms)
                }
        
        # Sort by score and return top 5
        sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for condition_name, diagnosis_info in sorted_conditions[:5]:
            diagnoses.append(diagnosis_info)
        
        return diagnoses
    
    def _generate_treatment_recommendations(self, search_results: Dict) -> List[Dict]:
        """Generate treatment recommendations based on search results"""
        recommendations = []
        
        # Extract treatments from medical conditions
        for condition in search_results.get('medical_conditions', []):
            treatments = condition.get('treatments', [])
            for treatment in treatments:
                recommendations.append({
                    'treatment': treatment,
                    'condition': condition.get('condition', ''),
                    'source': 'medical_condition',
                    'evidence_level': 'clinical_guideline'
                })
        
        # Extract drug information
        for drug in search_results.get('drugs', []):
            recommendations.append({
                'treatment': drug.get('drug_name', ''),
                'indications': drug.get('indications', []),
                'source': 'drug_database',
                'evidence_level': 'pharmaceutical_data'
            })
        
        return recommendations[:10]  # Limit to top 10
    
    def _extract_research_sources(self, search_results: Dict) -> List[Dict]:
        """Extract research sources from search results"""
        sources = []
        
        for paper in search_results.get('research_papers', []):
            source = {
                'title': paper.get('title', ''),
                'authors': paper.get('authors', ''),
                'journal': paper.get('journal', ''),
                'year': paper.get('year', ''),
                'pmid': paper.get('pmid', ''),
                'doi': paper.get('doi', ''),
                'full_text_url': paper.get('full_text_url', ''),
                'pubmed_url': paper.get('pubmed_url', ''),
                'evidence_level': paper.get('evidence_level', ''),
                'clinical_significance': paper.get('clinical_significance', '')
            }
            sources.append(source)
        
        return sources[:5]  # Limit to top 5
    
    def _calculate_confidence_score(self, symptoms: List[str], search_results: Dict) -> float:
        """Calculate confidence score for the analysis"""
        if not symptoms:
            return 0.3
        
        total_results = search_results.get('total_results', 0)
        symptom_count = len(symptoms)
        
        # Base confidence on number of symptoms and search results
        confidence = min(0.3 + (symptom_count * 0.1) + (total_results * 0.05), 0.95)
        
        return round(confidence, 2)
    
    def get_system_status(self) -> Dict:
        """Get system status and statistics"""
        return {
            'status': 'operational',
            'medical_conditions': len(self.medical_knowledge),
            'drugs': len(self.drug_database),
            'clinical_guidelines': len(self.clinical_guidelines),
            'research_papers': len(self.research_papers),
            'symptom_lexicon_entries': len(self.symptom_lexicon),
            'total_documents': self.loaded_documents,
            'memory_usage': 'optimized',
            'response_time': 'fast'
        }

# Global instance for easy access
_rag_instance = None

def get_lightweight_rag_system():
    """Get or create a singleton instance of the lightweight RAG system"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LightweightRAGSystem()
    return _rag_instance
