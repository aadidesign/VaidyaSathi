#!/usr/bin/env python
"""
Test script for personalized research paper recommendations
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cdss_project.settings')
django.setup()

from Rag.rag_system import get_rag_system
import json

def test_personalized_research():
    """Test personalized research paper recommendations with different patient profiles"""
    
    print("\n" + "="*80)
    print("TESTING PERSONALIZED RESEARCH PAPER RECOMMENDATIONS")
    print("="*80 + "\n")
    
    # Initialize RAG system
    print("📊 Initializing RAG system...")
    rag = get_rag_system()
    print("✅ RAG system initialized\n")
    
    # Test Case 1: Elderly patient with heart disease
    print("\n" + "-"*80)
    print("TEST CASE 1: Elderly Patient with Heart Disease")
    print("-"*80)
    
    query1 = "72 year old male with chest pain and shortness of breath"
    print(f"Query: {query1}")
    print("\nAnalyzing...")
    
    result1 = rag.analyze_case(query1)
    
    if result1.get('research_papers'):
        print(f"\n✅ Found research papers for {len(result1['research_papers'])} conditions")
        for condition, papers in result1['research_papers'].items():
            print(f"\n📚 Research papers for: {condition}")
            for i, paper in enumerate(papers[:2], 1):  # Show first 2 papers
                print(f"\n  Paper {i}:")
                print(f"    Title: {paper.get('title', 'N/A')}")
                print(f"    Year: {paper.get('year', 'N/A')}")
                
                # Check personalization
                if paper.get('is_personalized'):
                    print(f"    ✨ PERSONALIZED: Yes")
                    print(f"    Relevance: {paper.get('patient_relevance', 'N/A')}")
                else:
                    print(f"    ✨ PERSONALIZED: No (General research)")
                
                if paper.get('symptom_relevance'):
                    print(f"    🎯 Symptom Match: Yes")
                
                if paper.get('symptom_matches'):
                    print(f"    Matching Symptoms: {paper.get('symptom_matches')}")
    else:
        print("❌ No research papers found")
    
    # Test Case 2: Pediatric patient with asthma
    print("\n" + "-"*80)
    print("TEST CASE 2: Pediatric Patient with Asthma")
    print("-"*80)
    
    query2 = "8 year old child with wheezing, cough, and difficulty breathing"
    print(f"Query: {query2}")
    print("\nAnalyzing...")
    
    result2 = rag.analyze_case(query2)
    
    if result2.get('research_papers'):
        print(f"\n✅ Found research papers for {len(result2['research_papers'])} conditions")
        for condition, papers in result2['research_papers'].items():
            print(f"\n📚 Research papers for: {condition}")
            for i, paper in enumerate(papers[:2], 1):
                print(f"\n  Paper {i}:")
                print(f"    Title: {paper.get('title', 'N/A')}")
                print(f"    Year: {paper.get('year', 'N/A')}")
                
                if paper.get('is_personalized'):
                    print(f"    ✨ PERSONALIZED: Yes")
                    print(f"    Relevance: {paper.get('patient_relevance', 'N/A')}")
                else:
                    print(f"    ✨ PERSONALIZED: No (General research)")
                
                if paper.get('symptom_relevance'):
                    print(f"    🎯 Symptom Match: Yes")
                
                if paper.get('symptom_matches'):
                    print(f"    Matching Symptoms: {paper.get('symptom_matches')}")
    else:
        print("❌ No research papers found")
    
    # Test Case 3: Adult female with autoimmune symptoms
    print("\n" + "-"*80)
    print("TEST CASE 3: Adult Female with Autoimmune Symptoms")
    print("-"*80)
    
    query3 = "45 year old female with fatigue, joint pain, rash, and fever"
    print(f"Query: {query3}")
    print("\nAnalyzing...")
    
    result3 = rag.analyze_case(query3)
    
    if result3.get('research_papers'):
        print(f"\n✅ Found research papers for {len(result3['research_papers'])} conditions")
        for condition, papers in result3['research_papers'].items():
            print(f"\n📚 Research papers for: {condition}")
            for i, paper in enumerate(papers[:2], 1):
                print(f"\n  Paper {i}:")
                print(f"    Title: {paper.get('title', 'N/A')}")
                print(f"    Year: {paper.get('year', 'N/A')}")
                
                if paper.get('is_personalized'):
                    print(f"    ✨ PERSONALIZED: Yes")
                    print(f"    Relevance: {paper.get('patient_relevance', 'N/A')}")
                else:
                    print(f"    ✨ PERSONALIZED: No (General research)")
                
                if paper.get('symptom_relevance'):
                    print(f"    🎯 Symptom Match: Yes")
                
                if paper.get('symptom_matches'):
                    print(f"    Matching Symptoms: {paper.get('symptom_matches')}")
    else:
        print("❌ No research papers found")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_conditions = 0
    total_papers = 0
    total_personalized = 0
    total_symptom_matches = 0
    
    for result in [result1, result2, result3]:
        if result.get('research_papers'):
            for condition, papers in result['research_papers'].items():
                total_conditions += 1
                total_papers += len(papers)
                for paper in papers:
                    if paper.get('is_personalized'):
                        total_personalized += 1
                    if paper.get('symptom_relevance'):
                        total_symptom_matches += 1
    
    print(f"\n✅ Total Conditions Analyzed: {total_conditions}")
    print(f"📚 Total Research Papers Retrieved: {total_papers}")
    print(f"✨ Personalized Papers: {total_personalized} ({(total_personalized/total_papers*100) if total_papers > 0 else 0:.1f}%)")
    print(f"🎯 Papers with Symptom Matches: {total_symptom_matches} ({(total_symptom_matches/total_papers*100) if total_papers > 0 else 0:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ PERSONALIZED RESEARCH PAPER SYSTEM TEST COMPLETE")
    print("="*80 + "\n")
    
    # Verify personalization is working
    if total_personalized > 0:
        print("✅ SUCCESS: Personalization is working correctly!")
        print("   Research papers are being customized based on patient context.")
        return True
    else:
        print("⚠️  WARNING: No personalized papers found.")
        print("   This might be expected if the database doesn't have matching papers.")
        print("   The system should still add personalization metadata.")
        return False

if __name__ == '__main__':
    try:
        success = test_personalized_research()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

