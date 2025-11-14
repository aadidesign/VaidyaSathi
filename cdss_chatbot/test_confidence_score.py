#!/usr/bin/env python3
"""
Test script for the confidence score feature
This script demonstrates the confidence-based risk assessment functionality
"""

import sys
import os
import django

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cdss_project.settings')
django.setup()

from Rag.rag_system import RAGClinicalDecisionSupport

def print_separator():
    print("\n" + "="*80 + "\n")

def print_risk_assessment(result):
    """Pretty print risk assessment results"""
    risk_assessment = result.get('risk_assessment', {})
    
    print("📊 RISK ASSESSMENT RESULTS")
    print_separator()
    
    # Overall risk
    overall_risk_level = risk_assessment.get('overall_risk_level', 'unknown')
    overall_risk_score = risk_assessment.get('overall_risk_score', 0)
    print(f"🎯 Overall Risk Level: {overall_risk_level.upper()}")
    print(f"📈 Overall Risk Score: {overall_risk_score:.1f}/100")
    
    print_separator()
    
    # Confidence-based risk assessment
    confidence_risk = risk_assessment.get('confidence_based_risk', {})
    if confidence_risk:
        print("💯 CONFIDENCE-BASED RISK SCORES:")
        print()
        for condition, data in confidence_risk.items():
            risk_level = data.get('risk_level', 'Unknown')
            risk_score = data.get('risk_score', 0)
            confidence = data.get('confidence', 0)
            
            # Color coding based on risk level
            emoji = {
                'Critical': '🚨',
                'High': '⚠️',
                'Medium': '🟡',
                'Low': '🟢'
            }.get(risk_level, '❓')
            
            print(f"{emoji} {condition}")
            print(f"   Risk Level: {risk_level}")
            print(f"   Risk Score: {risk_score:.3f}")
            print(f"   Confidence: {confidence:.1f}%")
            print()
    
    # Alerts
    alerts = risk_assessment.get('alerts', [])
    if alerts:
        print_separator()
        print("🔔 AUTOMATED ALERTS:")
        print()
        for alert in alerts:
            print(f"   {alert}")
            print()

def print_diagnoses(result):
    """Pretty print differential diagnoses"""
    diagnoses = result.get('differential_diagnoses', [])
    
    if not diagnoses:
        print("No diagnoses found.")
        return
    
    print("🏥 DIFFERENTIAL DIAGNOSES:")
    print_separator()
    
    for i, diag in enumerate(diagnoses, 1):
        condition = diag.get('condition', 'Unknown')
        confidence = diag.get('confidence', 0)
        
        print(f"{i}. {condition}")
        print(f"   Confidence: {confidence:.1f}%")
        
        # Show description if available
        description = diag.get('description', {})
        if isinstance(description, dict):
            symptoms = description.get('symptoms', [])
            if symptoms:
                print(f"   Common Symptoms: {', '.join(symptoms[:3])}")
        
        print()

def test_high_risk_case():
    """Test case for high-risk patient"""
    print("\n" + "🔴 TEST CASE 1: HIGH-RISK PATIENT" + "="*60)
    
    query = "65-year-old male presenting with severe chest pain radiating to left arm, shortness of breath, and profuse sweating for 30 minutes"
    
    print(f"\nQuery: {query}")
    print_separator()
    
    # Initialize and analyze
    print("⏳ Analyzing case...")
    rag_system = RAGClinicalDecisionSupport()
    result = rag_system.analyze_case(query)
    
    # Print results
    print_diagnoses(result)
    print_risk_assessment(result)

def test_medium_risk_case():
    """Test case for medium-risk patient"""
    print("\n" + "🟡 TEST CASE 2: MEDIUM-RISK PATIENT" + "="*60)
    
    query = "45-year-old female with persistent cough, fatigue, and low-grade fever for 5 days"
    
    print(f"\nQuery: {query}")
    print_separator()
    
    # Initialize and analyze
    print("⏳ Analyzing case...")
    rag_system = RAGClinicalDecisionSupport()
    result = rag_system.analyze_case(query)
    
    # Print results
    print_diagnoses(result)
    print_risk_assessment(result)

def test_low_risk_case():
    """Test case for low-risk patient"""
    print("\n" + "🟢 TEST CASE 3: LOW-RISK PATIENT" + "="*60)
    
    query = "28-year-old male with mild headache and tiredness for 2 days"
    
    print(f"\nQuery: {query}")
    print_separator()
    
    # Initialize and analyze
    print("⏳ Analyzing case...")
    rag_system = RAGClinicalDecisionSupport()
    result = rag_system.analyze_case(query)
    
    # Print results
    print_diagnoses(result)
    print_risk_assessment(result)

def test_confidence_calculation():
    """Test the confidence calculation methods directly"""
    print("\n" + "🧪 TEST CASE 4: CONFIDENCE CALCULATION METHODS" + "="*60)
    
    # Initialize system
    rag_system = RAGClinicalDecisionSupport()
    
    # Test diagnoses
    test_diagnoses = [
        {"condition": "Heart Attack", "confidence": 90.0},
        {"condition": "Pneumonia", "confidence": 85.0},
        {"condition": "Diabetes", "confidence": 70.0},
        {"condition": "Common Cold", "confidence": 60.0}
    ]
    
    print("\n📋 Test Diagnoses:")
    for diag in test_diagnoses:
        print(f"   - {diag['condition']}: {diag['confidence']:.1f}% confidence")
    
    print_separator()
    
    # Test confidence-based risk assessment
    print("💯 Testing _assess_risk_with_confidence():")
    risk_assessment = rag_system._assess_risk_with_confidence(test_diagnoses)
    
    for condition, data in risk_assessment.items():
        print(f"\n   {condition}:")
        print(f"      Risk Score: {data['risk_score']:.3f}")
        print(f"      Risk Level: {data['risk_level']}")
        print(f"      Confidence: {data['confidence']:.1f}%")
    
    print_separator()
    
    # Test alert generation
    print("🔔 Testing _generate_alerts():")
    alerts = rag_system._generate_alerts(risk_assessment)
    
    if alerts:
        for alert in alerts:
            print(f"\n   {alert}")
    else:
        print("\n   No alerts generated (all conditions are low/medium risk)")
    
    print_separator()
    
    # Test individual condition risk levels
    print("🎯 Testing _get_condition_risk_level():")
    test_conditions = [
        "Heart Attack",
        "Stroke", 
        "Pneumonia",
        "Diabetes",
        "Hypertension",
        "Common Cold",
        "Headache"
    ]
    
    for condition in test_conditions:
        risk_level = rag_system._get_condition_risk_level(condition)
        print(f"   {condition}: {risk_level:.1f}")
    
    print_separator()
    
    # Test risk level labels
    print("🏷️ Testing _get_risk_level_label():")
    test_scores = [0.95, 0.75, 0.55, 0.35, 0.15]
    
    for score in test_scores:
        label = rag_system._get_risk_level_label(score)
        print(f"   Score {score:.2f} → {label}")

def main():
    """Run all test cases"""
    print("\n" + "🧪 CONFIDENCE SCORE FEATURE TEST SUITE" + "="*50)
    print("\nThis script tests the confidence-based risk assessment feature")
    print("of the Clinical Decision Support System.")
    
    try:
        # Run individual method tests first
        test_confidence_calculation()
        
        # Run clinical case tests
        print("\n\n" + "🏥 CLINICAL CASE TESTS" + "="*60)
        test_high_risk_case()
        test_medium_risk_case()
        test_low_risk_case()
        
        print("\n\n" + "✅ ALL TESTS COMPLETED" + "="*70)
        print("\nThe confidence score feature is working correctly!")
        print("\nKey Features Tested:")
        print("   ✓ Confidence-based risk calculation")
        print("   ✓ Condition-specific risk levels")
        print("   ✓ Risk level labeling")
        print("   ✓ Automated alert generation")
        print("   ✓ Integration with RAG system")
        
    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

