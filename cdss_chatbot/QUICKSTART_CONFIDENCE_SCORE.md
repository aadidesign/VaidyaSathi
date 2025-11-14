# 🚀 Quick Start Guide - Confidence Score Feature

## 1-Minute Setup

### ✅ Step 1: Verify Installation
The confidence score feature is already integrated! No additional installation needed.

```bash
# Verify FAISS is installed
pip list | grep faiss-cpu
# Should show: faiss-cpu 1.8.0 or higher
```

### ✅ Step 2: Test the Feature (30 seconds)

```bash
cd cdss_chatbot
python test_confidence_score.py
```

**Expected Output:**
```
🧪 CONFIDENCE SCORE FEATURE TEST SUITE
================================================================================

🧪 TEST CASE 4: CONFIDENCE CALCULATION METHODS
================================================================================

💯 Testing _assess_risk_with_confidence():

   Heart Attack:
      Risk Score: 0.900
      Risk Level: Critical
      Confidence: 90.0%

   Pneumonia:
      Risk Score: 0.850
      Risk Level: Critical
      Confidence: 85.0%

🔔 Testing _generate_alerts():

   🚨 CRITICAL ALERT: Heart Attack detected with 90.0% confidence - Seek immediate emergency medical attention!
   🚨 CRITICAL ALERT: Pneumonia detected with 85.0% confidence - Seek immediate emergency medical attention!

✅ ALL TESTS COMPLETED
```

### ✅ Step 3: Use in Your Code (2 minutes)

```python
from Rag.rag_system import RAGClinicalDecisionSupport

# Initialize
rag = RAGClinicalDecisionSupport()

# Analyze
result = rag.analyze_case("65-year-old male with chest pain")

# Access confidence-based risk
risk = result['risk_assessment']['confidence_based_risk']
alerts = result['risk_assessment']['alerts']

# Display
for condition, data in risk.items():
    print(f"{condition}: {data['risk_level']} ({data['confidence']:.1f}% confidence)")

for alert in alerts:
    print(alert)
```

## 📊 Feature Overview

### What It Does
- ✅ Calculates risk scores based on AI confidence + condition severity
- ✅ Generates automatic alerts for high-risk conditions
- ✅ Provides 4-tier risk levels (Critical/High/Medium/Low)
- ✅ Works seamlessly with existing CDSS functionality

### Risk Calculation Formula
```
Risk Score = (AI Confidence / 100) × Condition Risk Level

Examples:
- Heart Attack (90% confidence, 1.0 risk) = 0.90 → Critical
- Diabetes (70% confidence, 0.7 risk) = 0.49 → Medium
- Headache (60% confidence, 0.5 risk) = 0.30 → Low
```

## 🎯 Quick Examples

### Example 1: High-Risk Case
```python
query = "Severe chest pain with arm numbness and sweating"
result = rag.analyze_case(query)

# Output includes:
# - Diagnosis: "Heart Attack" (90% confidence)
# - Risk Level: "Critical"
# - Alert: "🚨 CRITICAL ALERT: Seek immediate emergency medical attention!"
```

### Example 2: Medium-Risk Case
```python
query = "Persistent cough and mild fever for 3 days"
result = rag.analyze_case(query)

# Output includes:
# - Diagnosis: "Bronchitis" (75% confidence)  
# - Risk Level: "Medium"
# - No critical alerts
```

### Example 3: API Usage
```bash
# Start server
python manage.py runserver

# Test API
curl -X POST http://127.0.0.1:8000/api/rag-chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Patient with severe headache and confusion"}'

# Response includes confidence_based_risk and alerts
```

## 📋 Risk Level Guide

| Level | Score | Icon | Action |
|-------|-------|------|--------|
| **Critical** | ≥0.8 | 🚨 | Emergency attention NOW |
| **High** | 0.6-0.79 | ⚠️ | Urgent evaluation needed |
| **Medium** | 0.4-0.59 | 🟡 | Schedule consultation |
| **Low** | <0.4 | 🟢 | Routine follow-up |

## 🔍 What's in the Response?

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
      "🚨 CRITICAL ALERT: Heart Attack detected..."
    ]
  }
}
```

## 💡 Pro Tips

1. **Check Alerts First**: Always check `risk_assessment['alerts']` for urgent conditions
2. **Use Risk Scores**: Sort by `risk_score` to prioritize diagnoses
3. **Combine Assessments**: Use both `confidence_based_risk` and `overall_risk_level`
4. **Customize Risk Levels**: Modify `_get_condition_risk_level()` for your needs

## 🛠️ Customization

### Adjust Risk Levels
Edit `cdss_chatbot/Rag/rag_system.py`:

```python
def _get_condition_risk_level(self, condition: str) -> float:
    # Add your custom conditions
    my_high_risk_conditions = ["Sepsis", "Meningitis"]
    
    if any(c in condition.lower() for c in my_high_risk_conditions):
        return 1.0  # Critical
    
    # Default behavior
    return 0.5
```

### Modify Alert Messages
Edit the `_generate_alerts()` method:

```python
def _generate_alerts(self, risk_assessment: Dict) -> List[str]:
    alerts = []
    for condition, assessment in risk_assessment.items():
        if assessment['risk_level'] == "Critical":
            # Customize your alert message here
            alerts.append(f"URGENT: {condition} requires immediate care!")
    return alerts
```

## 📚 Full Documentation

- **Complete Guide**: [CONFIDENCE_SCORE_FEATURE.md](CONFIDENCE_SCORE_FEATURE.md)
- **Integration Details**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
- **Test Script**: [test_confidence_score.py](test_confidence_score.py)

## ❓ Troubleshooting

### Issue: ImportError
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

### Issue: No alerts generated
```python
# Check if risk assessment exists
if 'alerts' in result['risk_assessment']:
    print(result['risk_assessment']['alerts'])
else:
    print("No high-risk conditions detected")
```

### Issue: Test script fails
```bash
# Ensure Django is set up
cd cdss_chatbot
python manage.py migrate
export DJANGO_SETTINGS_MODULE=cdss_project.settings

# Run test again
python test_confidence_score.py
```

## 🎉 You're Ready!

The confidence score feature is fully integrated and ready to use. Start by running the test script, then integrate it into your workflow!

**Next Steps:**
1. ✅ Run `python test_confidence_score.py`
2. ✅ Test with your own queries
3. ✅ Customize risk levels if needed
4. ✅ Deploy to production (with medical review)

---

**Need Help?** See [CONFIDENCE_SCORE_FEATURE.md](CONFIDENCE_SCORE_FEATURE.md) for detailed documentation.

