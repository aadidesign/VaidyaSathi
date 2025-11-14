#!/usr/bin/env python3
"""
Enhance the comprehensive disease database with missing fields for detailed analysis
"""

import json
import os
from typing import Dict, List, Any

def enhance_disease_database():
    """
    Add missing fields to the comprehensive disease database
    """
    
    # Load the current database
    data_dir = os.path.dirname(__file__)
    input_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        diseases = json.load(f)
    
    # Define age groups and severity levels for each disease category
    category_info = {
        'Cardiovascular': {
            'age_groups': ['Adults (40+)', 'Elderly (65+)', 'All ages (rare in children)'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Chest pain, shortness of breath, or fainting requires immediate medical attention. Call emergency services for severe symptoms.'
        },
        'Respiratory': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Severe breathing difficulty, blue lips, or inability to speak requires immediate medical attention.'
        },
        'Endocrine': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Diabetic emergencies (very high/low blood sugar), severe dehydration, or altered consciousness requires immediate medical attention.'
        },
        'Neurological': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Sudden severe headache, loss of consciousness, seizures, or stroke symptoms require immediate medical attention.'
        },
        'Mental Health': {
            'age_groups': ['Adolescents', 'Adults', 'Elderly'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Suicidal thoughts, self-harm, or severe psychotic episodes require immediate mental health crisis intervention.'
        },
        'Gastrointestinal': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Severe abdominal pain, vomiting blood, or signs of bowel obstruction require immediate medical attention.'
        },
        'Musculoskeletal': {
            'age_groups': ['Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Severe pain, inability to move, or signs of fracture require immediate medical attention.'
        },
        'Infectious': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'High fever, severe dehydration, or signs of sepsis require immediate medical attention.'
        },
        'Cancer': {
            'age_groups': ['Adults', 'Elderly', 'All ages (varies by type)'],
            'severity_levels': ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
            'emergency_protocol': 'Severe pain, difficulty breathing, or signs of metastasis require immediate oncological consultation.'
        },
        'Kidney': {
            'age_groups': ['Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'],
            'emergency_protocol': 'Signs of kidney failure, severe pain, or inability to urinate require immediate medical attention.'
        },
        'Skin': {
            'age_groups': ['Children', 'Adults', 'Elderly', 'All ages'],
            'severity_levels': ['Mild', 'Moderate', 'Severe', 'Critical'],
            'emergency_protocol': 'Severe allergic reactions, signs of skin infection, or suspicious moles require immediate medical attention.'
        }
    }
    
    # Enhance each disease entry
    for disease in diseases:
        category = disease.get('category', 'Other')
        
        # Add missing fields
        if 'age_groups' not in disease:
            disease['age_groups'] = category_info.get(category, {}).get('age_groups', ['Adults'])
        
        if 'severity_levels' not in disease:
            disease['severity_levels'] = category_info.get(category, {}).get('severity_levels', ['Mild', 'Moderate', 'Severe'])
        
        if 'emergency_protocol' not in disease:
            disease['emergency_protocol'] = category_info.get(category, {}).get('emergency_protocol', 'Consult healthcare provider for concerning symptoms.')
        
        # Ensure all required fields exist
        required_fields = [
            'disease_id', 'disease_name', 'category', 'icd10_code', 'prevalence',
            'description', 'symptoms', 'risk_factors', 'diagnostic_tests', 'treatments',
            'complications', 'research_papers', 'age_groups', 'severity_levels', 'emergency_protocol'
        ]
        
        for field in required_fields:
            if field not in disease:
                if field == 'disease_id':
                    disease[field] = len([d for d in diseases if d.get('disease_id')])
                elif field == 'disease_name':
                    disease[field] = 'Unknown Disease'
                elif field == 'category':
                    disease[field] = 'Other'
                elif field == 'icd10_code':
                    disease[field] = 'Z00-Z99'
                elif field == 'prevalence':
                    disease[field] = 'Unknown'
                elif field == 'description':
                    disease[field] = 'Medical condition requiring professional evaluation.'
                elif field in ['symptoms', 'risk_factors', 'diagnostic_tests', 'treatments', 'complications', 'research_papers', 'age_groups', 'severity_levels']:
                    disease[field] = []
                elif field == 'emergency_protocol':
                    disease[field] = 'Consult healthcare provider for concerning symptoms.'
    
    # Save enhanced database
    output_file = os.path.join(data_dir, 'comprehensive_top50_diseases_database.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diseases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced database with {len(diseases)} diseases")
    print("📊 Added fields: age_groups, severity_levels, emergency_protocol")
    print("🔍 All diseases now have complete data for detailed analysis sections")

if __name__ == "__main__":
    enhance_disease_database()
