# 🏗️ Confidence Score Feature - System Architecture

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                   │
│              "65-year-old male with chest pain"                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    RAGClinicalDecisionSupport                        │
│                      analyze_case(query)                             │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    ┌───────────────────────┐
                    │  NLP Processing       │
                    │  Symptom Extraction   │
                    │  Semantic Analysis    │
                    └───────────────────────┘
                                ↓
                    ┌───────────────────────┐
                    │  RAG Retrieval        │
                    │  FAISS Vector Search  │
                    │  Context Generation   │
                    └───────────────────────┘
                                ↓
                    ┌───────────────────────┐
                    │  Diagnosis Generation │
                    │  (Gemini AI)          │
                    │  + Confidence Scores  │
                    └───────────────────────┘
                                ↓
        ┌───────────────────────────────────────────────┐
        │         CONFIDENCE SCORE FEATURE              │
        │                                               │
        │  ┌─────────────────────────────────────────┐ │
        │  │ 1. _assess_risk_with_confidence()       │ │
        │  │    - Input: Diagnoses with confidence   │ │
        │  │    - Calculates risk scores             │ │
        │  └─────────────────────────────────────────┘ │
        │                    ↓                          │
        │  ┌─────────────────────────────────────────┐ │
        │  │ 2. _get_condition_risk_level()          │ │
        │  │    - Input: Condition name              │ │
        │  │    - Returns: 1.0 / 0.7 / 0.5           │ │
        │  └─────────────────────────────────────────┘ │
        │                    ↓                          │
        │  ┌─────────────────────────────────────────┐ │
        │  │ 3. _get_risk_level_label()              │ │
        │  │    - Input: Risk score (0.0-1.0)        │ │
        │  │    - Returns: Critical/High/Med/Low     │ │
        │  └─────────────────────────────────────────┘ │
        │                    ↓                          │
        │  ┌─────────────────────────────────────────┐ │
        │  │ 4. _generate_alerts()                   │ │
        │  │    - Input: Risk assessment             │ │
        │  │    - Generates: 🚨 Critical alerts      │ │
        │  │                 ⚠️ High-risk alerts     │ │
        │  └─────────────────────────────────────────┘ │
        └───────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        API Response                                  │
│  {                                                                   │
│    "differential_diagnoses": [...],                                 │
│    "risk_assessment": {                                             │
│      "overall_risk_level": "high",                                  │
│      "confidence_based_risk": {                                     │
│        "Heart Attack": {                                            │
│          "risk_score": 0.900,                                       │
│          "risk_level": "Critical",                                  │
│          "confidence": 90.0                                         │
│        }                                                            │
│      },                                                             │
│      "alerts": ["🚨 CRITICAL ALERT: ..."]                          │
│    }                                                                │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### Step 1: Diagnosis with Confidence
```python
Input:
{
  "condition": "Heart Attack",
  "confidence": 90.0,
  "description": {...}
}
```

### Step 2: Risk Score Calculation
```python
confidence = 90.0 / 100  # 0.90
condition_risk_level = 1.0  # Heart Attack is critical
risk_score = 0.90 × 1.0 = 0.900
```

### Step 3: Risk Level Labeling
```python
if risk_score >= 0.8:
    risk_level = "Critical"
elif risk_score >= 0.6:
    risk_level = "High"
elif risk_score >= 0.4:
    risk_level = "Medium"
else:
    risk_level = "Low"
```

### Step 4: Alert Generation
```python
if risk_level == "Critical":
    alert = "🚨 CRITICAL ALERT: Heart Attack detected with 90.0% confidence - Seek immediate emergency medical attention!"
elif risk_level == "High":
    alert = "⚠️ HIGH RISK ALERT: ... - Urgent medical evaluation required!"
else:
    alert = None  # No alert for medium/low risk
```

## 🎯 Risk Level Matrix

```
┌────────────────────────────────────────────────────────────────┐
│                    Risk Calculation Matrix                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Confidence (%)                                                │
│   100 ┤ 1.00│0.70│0.50│  🚨 Critical (≥0.8)                   │
│       │     │    │    │                                         │
│    80 ┤ 0.80│0.56│0.40│  ⚠️ High (0.6-0.79)                   │
│       │     │    │    │                                         │
│    60 ┤ 0.60│0.42│0.30│  🟡 Medium (0.4-0.59)                 │
│       │     │    │    │                                         │
│    40 ┤ 0.40│0.28│0.20│  🟢 Low (<0.4)                        │
│       │     │    │    │                                         │
│     0 ┤ 0.00│0.00│0.00│                                         │
│       └─────┴────┴────┘                                         │
│        1.0   0.7  0.5                                           │
│        Condition Risk Level                                     │
│                                                                 │
│   Legend:                                                       │
│   1.0 = Critical conditions (Heart Attack, Stroke)             │
│   0.7 = High-risk conditions (Diabetes, Hypertension)          │
│   0.5 = Moderate conditions (Other)                            │
└────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Integration

### Before Integration
```
analyze_case()
    ↓
  Generate Diagnoses (with confidence)
    ↓
  Perform Risk Assessment (existing)
    ↓
  Return Results
```

### After Integration
```
analyze_case()
    ↓
  Generate Diagnoses (with confidence)
    ↓
  Perform Risk Assessment (existing)
    ↓
  ┌─────────────────────────────────┐
  │ NEW: Confidence-Based Risk      │
  │  1. _assess_risk_with_confidence│
  │  2. _generate_alerts            │
  │  3. Merge with risk_assessment  │
  └─────────────────────────────────┘
    ↓
  Return Enhanced Results
```

## 📦 Module Structure

```
cdss_chatbot/Rag/
├── rag_system.py
│   ├── Class: RAGClinicalDecisionSupport
│   │   ├── __init__()
│   │   ├── analyze_case()  ← Entry point
│   │   │   └── Calls confidence score methods
│   │   │
│   │   ├── [NEW] _assess_risk_with_confidence()
│   │   │   └── Calculates risk scores for all diagnoses
│   │   │
│   │   ├── [NEW] _get_condition_risk_level()
│   │   │   └── Returns 1.0 / 0.7 / 0.5 based on condition
│   │   │
│   │   ├── [NEW] _get_risk_level_label()
│   │   │   └── Converts score to Critical/High/Med/Low
│   │   │
│   │   └── [NEW] _generate_alerts()
│   │       └── Generates alert messages
│   │
│   └── [Existing methods remain unchanged]
```

## 🔀 Method Call Flow

```
User → analyze_case(query)
         │
         ├─→ Extract patient info
         ├─→ Retrieve medical knowledge (FAISS)
         ├─→ Generate differential diagnoses (Gemini AI)
         │    └─→ Returns: [{condition, confidence, ...}, ...]
         │
         ├─→ _perform_risk_assessment()  [Existing]
         │    └─→ Returns: {overall_risk_level, overall_risk_score, ...}
         │
         ├─→ _assess_risk_with_confidence()  [NEW]
         │    │
         │    ├─→ For each diagnosis:
         │    │    ├─→ Get confidence (e.g., 90.0)
         │    │    ├─→ Call _get_condition_risk_level(condition)
         │    │    │    └─→ Returns: 1.0 or 0.7 or 0.5
         │    │    ├─→ Calculate: risk_score = confidence × risk_level
         │    │    └─→ Call _get_risk_level_label(risk_score)
         │    │         └─→ Returns: "Critical" or "High" or "Medium" or "Low"
         │    │
         │    └─→ Returns: {condition: {risk_score, risk_level, confidence}}
         │
         ├─→ _generate_alerts(confidence_based_risk)  [NEW]
         │    │
         │    ├─→ For each condition:
         │    │    ├─→ If risk_level == "Critical":
         │    │    │    └─→ Add: "🚨 CRITICAL ALERT: ..."
         │    │    └─→ If risk_level == "High":
         │    │         └─→ Add: "⚠️ HIGH RISK ALERT: ..."
         │    │
         │    └─→ Returns: [alert1, alert2, ...]
         │
         ├─→ Merge alerts into risk_assessment
         ├─→ Add confidence_based_risk to risk_assessment
         │
         └─→ Return complete results
              └─→ {
                    patient_info,
                    differential_diagnoses,
                    risk_assessment: {
                      overall_risk_level,
                      overall_risk_score,
                      confidence_based_risk,  ← NEW
                      alerts                   ← NEW
                    },
                    ...
                  }
```

## 🎨 Risk Level Color Coding

```
┌──────────────────────────────────────────────────────────────┐
│  Risk Level │ Color  │ Icon │ Score Range │ Alert Type      │
├─────────────┼────────┼──────┼─────────────┼─────────────────┤
│  Critical   │ 🔴 Red  │  🚨  │  ≥ 0.8      │ CRITICAL ALERT  │
│  High       │ 🟠 Orange│ ⚠️  │ 0.6 - 0.79  │ HIGH RISK ALERT │
│  Medium     │ 🟡 Yellow│ 🟡  │ 0.4 - 0.59  │ (No alert)      │
│  Low        │ 🟢 Green │ 🟢  │  < 0.4      │ (No alert)      │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Example Scenarios

### Scenario 1: High Confidence + Critical Condition
```
Input:
  Condition: "Heart Attack"
  Confidence: 92%

Calculation:
  risk_level = 1.0 (critical condition)
  risk_score = 0.92 × 1.0 = 0.92

Output:
  risk_level: "Critical" (0.92 ≥ 0.8)
  alert: "🚨 CRITICAL ALERT: Heart Attack detected with 92.0% confidence..."
```

### Scenario 2: Medium Confidence + High-Risk Condition
```
Input:
  Condition: "Diabetes"
  Confidence: 68%

Calculation:
  risk_level = 0.7 (high-risk condition)
  risk_score = 0.68 × 0.7 = 0.476

Output:
  risk_level: "Medium" (0.4 ≤ 0.476 < 0.6)
  alert: None (no alert for medium risk)
```

### Scenario 3: High Confidence + Moderate Condition
```
Input:
  Condition: "Common Cold"
  Confidence: 85%

Calculation:
  risk_level = 0.5 (moderate condition)
  risk_score = 0.85 × 0.5 = 0.425

Output:
  risk_level: "Medium" (0.4 ≤ 0.425 < 0.6)
  alert: None
```

## 🔍 Performance Characteristics

```
┌──────────────────────────────────────────────────────────┐
│  Operation                    │ Time      │ Complexity   │
├───────────────────────────────┼───────────┼──────────────┤
│  _assess_risk_with_confidence │ ~1-5ms    │ O(n)         │
│  _get_condition_risk_level    │ ~0.1ms    │ O(1)         │
│  _get_risk_level_label        │ ~0.01ms   │ O(1)         │
│  _generate_alerts             │ ~1-2ms    │ O(n)         │
│                                │           │              │
│  Total Overhead per Query     │ ~2-10ms   │ O(n)         │
│  (n = number of diagnoses)    │           │              │
└──────────────────────────────────────────────────────────┘

Notes:
- No additional API calls
- No database queries
- All calculations are in-memory
- Negligible impact on overall response time
```

## 🎯 Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    System Integration                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (React)                                           │
│    ↓                                                         │
│  API Request → POST /api/rag-chat/                          │
│    ↓                                                         │
│  Django View (views.py)                                     │
│    ↓                                                         │
│  RAGClinicalDecisionSupport.analyze_case()                  │
│    ↓                                                         │
│  [Confidence Score Methods]  ← Integration here             │
│    ↓                                                         │
│  Return JSON Response                                        │
│    ↓                                                         │
│  Frontend Display                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📝 Configuration Options

```python
# Easily customizable in rag_system.py

# 1. Adjust condition risk levels
def _get_condition_risk_level(self, condition: str) -> float:
    # Add your conditions here
    my_critical_conditions = ["Sepsis", "Meningitis"]
    return 1.0 if any(c in condition.lower() for c in my_critical_conditions) else 0.5

# 2. Modify risk thresholds
def _get_risk_level_label(self, risk_score: float) -> str:
    # Adjust thresholds here
    if risk_score >= 0.85:  # Changed from 0.8
        return "Critical"
    # ... etc

# 3. Customize alert messages
def _generate_alerts(self, risk_assessment: Dict) -> List[str]:
    # Modify alert text here
    alerts.append(f"URGENT: {condition} - Call 911!")
```

---

**This architecture diagram shows how the confidence score feature integrates seamlessly with your existing CDSS system!**

