"""
Advanced Semantic Parsing Module for Medical CDSS

This module implements:
1. Medical Named Entity Recognition (NER) and Entity Linking (NEL) 
2. Word Sense Disambiguation (WSD) for medical contexts
3. Semantic Role Labeling (SRL) and Relationship Extraction
4. Integration with biomedical knowledge bases (UMLS concepts)

Uses SciSpacy models trained on biomedical literature for accurate medical NLP.
"""

import spacy
from spacy import displacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Optional, Set
import json
import logging

logger = logging.getLogger(__name__)

class MedicalSemanticParser:
    """Advanced semantic parser for medical text with NER, NEL, WSD, and relationship extraction."""
    
    def __init__(self):
        """Initialize the medical semantic parser with SciSpacy models."""
        self._init_models()
        self._init_medical_patterns()
        
    def _init_models(self):
        """Initialize SciSpacy biomedical models."""
        try:
            # Load biomedical spaCy model
            print("Loading SciSpacy biomedical model...")
            self.nlp = spacy.load("en_core_sci_sm")
            
            # Add entity linker for UMLS linking
            print("Adding UMLS entity linker...")
            self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
            
            # Get the linker component for direct access
            self.linker = self.nlp.get_pipe("scispacy_linker")
            
            print("✅ SciSpacy model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load SciSpacy model: {e}")
            # Fallback to basic spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.linker = None
                print("⚠️ Using fallback spaCy model (no biomedical features)")
            except:
                logger.error("No spaCy model available")
                self.nlp = None
                self.linker = None
    
    def _init_medical_patterns(self):
        """Initialize patterns for medical relationship extraction."""
        # Semantic role patterns for medical relationships
        self.medical_relations = {
            'treats': [
                r'(\w+)\s+(?:treats?|cures?|helps?|alleviates?)\s+(\w+)',
                r'(\w+)\s+(?:is|was)\s+(?:used|given|prescribed)\s+(?:for|to treat)\s+(\w+)',
                r'(?:treatment|therapy|medication)\s+(?:for|of)\s+(\w+)\s+(?:with|using)\s+(\w+)'
            ],
            'causes': [
                r'(\w+)\s+(?:causes?|leads? to|results? in|triggers?)\s+(\w+)',
                r'(\w+)\s+(?:is|was)\s+(?:caused by|due to|from)\s+(\w+)',
                r'(?:side effect|adverse reaction)\s+(?:of|from)\s+(\w+)\s+(?:is|includes?)\s+(\w+)'
            ],
            'symptoms_of': [
                r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(?:symptom|sign|manifestation)\s+(?:of|for)\s+(\w+)',
                r'(?:patient|individual)\s+(?:presents?|has|shows?)\s+(\w+)\s+(?:in|with|for)\s+(\w+)',
                r'(\w+)\s+(?:associated with|seen in|occurs? in)\s+(\w+)'
            ],
            'diagnosed_with': [
                r'(?:patient|individual)\s+(?:diagnosed with|has|suffers? from)\s+(\w+)',
                r'(?:diagnosis|condition)\s+(?:of|is)\s+(\w+)',
                r'(\w+)\s+(?:patient|case|individual)'
            ]
        }
        
        # Medical entity types from biomedical literature
        self.medical_entity_types = {
            'DISEASE': ['disease', 'disorder', 'syndrome', 'condition', 'illness'],
            'DRUG': ['drug', 'medication', 'medicine', 'pharmaceutical', 'therapy'],
            'SYMPTOM': ['symptom', 'sign', 'manifestation', 'complaint'],
            'ANATOMY': ['organ', 'tissue', 'body part', 'anatomical structure'],
            'PROCEDURE': ['procedure', 'surgery', 'operation', 'treatment', 'therapy'],
            'TEST': ['test', 'examination', 'assay', 'screening', 'diagnostic']
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract medical entities using SciSpacy NER with UMLS linking.
        
        Args:
            text: Input medical text
            
        Returns:
            Dictionary with entity types and their details including UMLS CUIs
        """
        if not self.nlp:
            return {}
            
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_,
                'confidence': getattr(ent, 'confidence', 1.0),
                'umls_cuis': [],
                'umls_concepts': []
            }
            
            # Add UMLS linking information if available
            if self.linker and hasattr(ent._, 'kb_ents'):
                for cui, score in ent._.kb_ents[:3]:  # Top 3 concepts
                    try:
                        concept_info = self.linker.kb.cui_to_entity.get(cui, {})
                        entity_info['umls_cuis'].append(cui)
                        entity_info['umls_concepts'].append({
                            'cui': cui,
                            'name': getattr(concept_info, 'canonical_name', '') if hasattr(concept_info, 'canonical_name') else str(concept_info.get('canonical_name', '') if isinstance(concept_info, dict) else ''),
                            'definition': getattr(concept_info, 'definition', '') if hasattr(concept_info, 'definition') else str(concept_info.get('definition', '') if isinstance(concept_info, dict) else ''),
                            'types': getattr(concept_info, 'types', []) if hasattr(concept_info, 'types') else (concept_info.get('types', []) if isinstance(concept_info, dict) else []),
                            'score': score
                        })
                    except Exception as e:
                        logger.warning(f"Error processing UMLS concept {cui}: {e}")
                        continue
            
            entities[ent.label_].append(entity_info)
        
        return dict(entities)
    
    def word_sense_disambiguation(self, text: str, target_words: List[str] = None) -> Dict[str, Dict]:
        """
        Perform Word Sense Disambiguation for medical contexts.
        
        Args:
            text: Input text
            target_words: Specific words to disambiguate (if None, auto-detect)
            
        Returns:
            Dictionary with disambiguated word senses
        """
        if not self.nlp:
            return {}
            
        doc = self.nlp(text)
        disambiguated = {}
        
        # Auto-detect ambiguous medical terms if not specified
        if target_words is None:
            target_words = []
            for token in doc:
                if token.text.upper() in ['MI', 'MS', 'RA', 'BP', 'HR', 'PE', 'CT', 'MRI']:
                    target_words.append(token.text)
        
        for word in target_words:
            # Find the word in context
            for token in doc:
                if token.text.lower() == word.lower():
                    context_window = []
                    
                    # Get surrounding context
                    start_idx = max(0, token.i - 5)
                    end_idx = min(len(doc), token.i + 6)
                    
                    for i in range(start_idx, end_idx):
                        if i != token.i:
                            context_window.append(doc[i].text)
                    
                    # Medical context disambiguation
                    medical_context = self._get_medical_context(token, doc)
                    
                    disambiguated[word] = {
                        'token': token.text,
                        'pos': token.pos_,
                        'lemma': token.lemma_,
                        'context': ' '.join(context_window),
                        'medical_sense': self._resolve_medical_sense(token.text, medical_context),
                        'confidence': medical_context.get('confidence', 0.5)
                    }
        
        return disambiguated
    
    def _get_medical_context(self, token, doc) -> Dict:
        """Analyze medical context around a token."""
        medical_indicators = []
        confidence = 0.5
        
        # Look for medical entities nearby
        for ent in doc.ents:
            if abs(ent.start - token.i) <= 3:  # Within 3 tokens
                medical_indicators.append(ent.label_)
                confidence += 0.2
        
        # Look for medical keywords
        medical_keywords = ['patient', 'diagnosis', 'treatment', 'symptom', 'disease', 'medication']
        for keyword in medical_keywords:
            if keyword in doc.text.lower():
                confidence += 0.1
        
        return {
            'indicators': medical_indicators,
            'confidence': min(confidence, 1.0)
        }
    
    def _resolve_medical_sense(self, word: str, context: Dict) -> str:
        """Resolve the medical sense of an ambiguous word."""
        word_upper = word.upper()
        
        # Common medical abbreviation disambiguation
        medical_senses = {
            'MI': {
                'cardiac': 'Myocardial Infarction',
                'default': 'Michigan/Medical Insurance'
            },
            'MS': {
                'neuro': 'Multiple Sclerosis',
                'default': 'Master of Science/Mississippi'
            },
            'RA': {
                'rheum': 'Rheumatoid Arthritis',
                'default': 'Right Atrium/Resident Assistant'
            },
            'BP': {
                'cardio': 'Blood Pressure',
                'default': 'British Petroleum'
            },
            'PE': {
                'pulm': 'Pulmonary Embolism',
                'default': 'Physical Education'
            }
        }
        
        if word_upper in medical_senses:
            # Check if medical context supports medical sense
            if context.get('confidence', 0) > 0.7:
                for sense_key, sense_value in medical_senses[word_upper].items():
                    if sense_key != 'default':
                        return sense_value
            return medical_senses[word_upper]['default']
        
        return word  # Return original if no disambiguation needed
    
    def extract_medical_relationships(self, text: str) -> List[Dict]:
        """
        Extract semantic relationships from medical text using pattern matching and dependency parsing.
        
        Args:
            text: Input medical text
            
        Returns:
            List of extracted relationships
        """
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        relationships = []
        
        # Pattern-based relationship extraction
        for relation_type, patterns in self.medical_relations.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        relationships.append({
                            'relation': relation_type,
                            'subject': groups[0],
                            'object': groups[1] if len(groups) > 1 else '',
                            'confidence': 0.8,
                            'method': 'pattern_matching',
                            'text_span': match.group()
                        })
        
        # Dependency-based relationship extraction
        dep_relations = self._extract_dependency_relationships(doc)
        relationships.extend(dep_relations)
        
        return relationships
    
    def _extract_dependency_relationships(self, doc) -> List[Dict]:
        """Extract relationships using dependency parsing."""
        relationships = []
        
        for token in doc:
            # Look for medical verbs that indicate relationships
            if token.pos_ == 'VERB' and token.lemma_ in ['treat', 'cause', 'diagnose', 'prescribe', 'indicate']:
                subject = None
                object = None
                
                # Find subject and object
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subject = child.text
                    elif child.dep_ in ['dobj', 'pobj']:
                        object = child.text
                
                if subject and object:
                    relationships.append({
                        'relation': token.lemma_,
                        'subject': subject,
                        'object': object,
                        'confidence': 0.7,
                        'method': 'dependency_parsing',
                        'text_span': f"{subject} {token.text} {object}"
                    })
        
        return relationships
    
    def semantic_role_labeling(self, text: str) -> List[Dict]:
        """
        Perform semantic role labeling to identify who did what to whom.
        
        Args:
            text: Input text
            
        Returns:
            List of semantic roles and their arguments
        """
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        semantic_roles = []
        
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == 'VERB':
                    roles = {
                        'predicate': token.lemma_,
                        'agent': [],      # Who performs the action
                        'patient': [],    # What receives the action  
                        'instrument': [], # How/with what
                        'location': [],   # Where
                        'time': []        # When
                    }
                    
                    # Extract arguments based on dependency relations
                    for child in token.children:
                        if child.dep_ == 'nsubj':
                            roles['agent'].append(child.text)
                        elif child.dep_ in ['dobj', 'pobj']:
                            roles['patient'].append(child.text)
                        elif child.dep_ == 'prep':
                            # Check preposition type for role assignment
                            prep_text = child.text.lower()
                            if prep_text in ['with', 'using', 'by']:
                                for prep_child in child.children:
                                    if prep_child.dep_ == 'pobj':
                                        roles['instrument'].append(prep_child.text)
                            elif prep_text in ['in', 'at', 'on']:
                                for prep_child in child.children:
                                    if prep_child.dep_ == 'pobj':
                                        roles['location'].append(prep_child.text)
                    
                    # Only add if we have meaningful roles
                    if any(roles[key] for key in ['agent', 'patient']):
                        semantic_roles.append(roles)
        
        return semantic_roles
    
    def comprehensive_analysis(self, text: str) -> Dict:
        """
        Perform comprehensive semantic analysis combining all features.
        
        Args:
            text: Input medical text
            
        Returns:
            Complete semantic analysis including NER, WSD, relationships, and SRL
        """
        try:
            analysis = {
                'input_text': text,
                'medical_entities': self.extract_medical_entities(text),
                'word_sense_disambiguation': self.word_sense_disambiguation(text),
                'medical_relationships': self.extract_medical_relationships(text),
                'semantic_roles': self.semantic_role_labeling(text),
                'summary': {}
            }
            
            # Generate summary statistics
            analysis['summary'] = {
                'total_entities': sum(len(entities) for entities in analysis['medical_entities'].values()),
                'entity_types': list(analysis['medical_entities'].keys()),
                'total_relationships': len(analysis['medical_relationships']),
                'disambiguated_terms': len(analysis['word_sense_disambiguation']),
                'semantic_predicates': len(analysis['semantic_roles'])
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'input_text': text,
                'error': str(e),
                'medical_entities': {},
                'word_sense_disambiguation': {},
                'medical_relationships': [],
                'semantic_roles': [],
                'summary': {}
            }

# Global instance for easy access
_semantic_parser = None

def get_semantic_parser() -> MedicalSemanticParser:
    """Get or create the global semantic parser instance."""
    global _semantic_parser
    if _semantic_parser is None:
        _semantic_parser = MedicalSemanticParser()
    return _semantic_parser

def analyze_medical_semantics(text: str) -> Dict:
    """Convenience function for semantic analysis."""
    parser = get_semantic_parser()
    return parser.comprehensive_analysis(text)
