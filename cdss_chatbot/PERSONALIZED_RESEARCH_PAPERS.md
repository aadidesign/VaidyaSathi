# Personalized Research Paper Recommendations

## Overview

The Clinical Decision Support System now provides **real-time, patient-specific research paper recommendations** instead of generic templated information. Research papers are dynamically filtered, ranked, and personalized based on each patient's unique clinical presentation.

## What Changed

### Before
- Research papers were retrieved based only on the diagnosed condition name
- No consideration of patient demographics (age, gender)
- No filtering based on patient symptoms
- Generic recommendations that might not be relevant to the specific patient
- Same papers shown for all patients with the same condition

### After
- **Multi-factor personalization** using patient age, gender, symptoms, and clinical query
- **Semantic matching** using PubMedBERT to find papers most relevant to patient presentation
- **Symptom-based ranking** that prioritizes papers discussing the patient's specific symptoms
- **Age-specific filtering** (pediatric, adult, geriatric research)
- **Gender-specific considerations** when relevant
- **Personalized relevance explanations** showing why each paper is relevant to the patient

## Key Features

### 1. Patient-Specific Context Integration

The system now passes comprehensive patient context to the research paper retrieval:

```python
search_context = {
    'condition': condition,
    'age': patient_age,
    'gender': patient_gender,
    'symptoms': patient_symptoms,
    'clinical_query': clinical_query,
    'confidence': diagnosis.get('confidence', 0)
}
```

### 2. Enhanced PubMed Retrieval

**Symptom Matching:**
- Searches paper titles and abstracts for patient symptoms
- Calculates relevance score: +2.0 points per matching symptom
- Tracks number of symptom matches per paper

**Age-Specific Filtering:**
- For patients < 18: Searches for "pediatric", "children", "adolescent" keywords
- For patients > 65: Searches for "elderly", "geriatric", "older adults" keywords
- For adults 18-65: Searches for "adult", "middle-aged" keywords
- Adds +1.5 relevance points for age-appropriate papers

**Gender-Specific Filtering:**
- Matches patient gender in paper content
- Adds +1.0 relevance points for gender-specific research

**Smart Sorting:**
Papers are ranked by:
1. Patient relevance score (highest priority)
2. Publication year (most recent)
3. Clinical significance (High > Medium > Low)

### 3. PubMedBERT Semantic Enhancement

For larger paper sets, the system uses PubMedBERT to:
- Generate semantic embeddings of patient context and research papers
- Calculate cosine similarity between patient presentation and paper content
- Boost scores for papers mentioning patient symptoms
- Re-rank papers by semantic relevance

Example query construction:
```python
condition_query = f"{condition} medical research treatment diagnosis"

# Add symptoms
if symptoms:
    symptoms_text = ' '.join(symptoms[:5])
    condition_query += f" symptoms: {symptoms_text}"

# Add age context
if age < 18:
    condition_query += " pediatric children adolescent"
elif age > 65:
    condition_query += " elderly geriatric older adults"

# Add gender
if gender:
    condition_query += f" {gender}"
```

### 4. Personalization Metadata

Each research paper now includes:

```python
{
    'title': '...',
    'authors': '...',
    'journal': '...',
    'year': 2024,
    'summary': '...',
    
    # NEW: Personalization fields
    'patient_relevance': 'Discusses symptoms relevant to your case: chest pain, shortness of breath; Focuses on elderly patients, matching your age group',
    'is_personalized': True,
    'relevance_score': 8.5,
    'symptom_relevance': True,
    'symptom_matches': 3,
    'patient_relevance_score': 7.5
}
```

### 5. Enhanced UI Display

The frontend now highlights personalized papers with:

- **✨ Star indicator** for personalized papers
- **Personalized banner** showing why the paper is relevant
- **"Personalized" badge** for quick identification
- **"Symptom Match" badge** when paper discusses patient symptoms
- **Enhanced border and shadow** for personalized papers
- **Detailed relevance explanation** in green highlight box

Example relevance messages:
- "Discusses symptoms relevant to your case: chest pain, fatigue"
- "Focuses on elderly patients, matching your age group"
- "Includes gender-specific research relevant to female patients"

## Technical Implementation

### Backend Changes (rag_system.py)

#### 1. Updated Method Signatures

```python
# Before
def _retrieve_research_papers(self, diagnoses: List[Dict]) -> Dict[str, List[Dict]]

# After
def _retrieve_research_papers(
    self, 
    diagnoses: List[Dict], 
    patient_info: Dict = None, 
    clinical_query: str = ""
) -> Dict[str, List[Dict]]
```

#### 2. New Methods

**`_personalize_research_papers()`**
- Analyzes each paper for patient-specific relevance
- Generates personalized relevance explanations
- Adds metadata for UI display

**Enhanced `_retrieve_actual_pubmed_papers()`**
- Accepts search_context parameter
- Calculates patient-specific relevance scores
- Filters by symptoms, age, and gender
- Sorts by personalized relevance

**Enhanced `_enhance_research_papers_with_pubmedbert()`**
- Uses patient context in semantic queries
- Boosts scores for symptom matches
- Considers age and gender in similarity calculation

**Enhanced `_generate_synthetic_research_papers()`**
- Includes patient context in LLM prompts
- Generates age and symptom-specific fallback papers
- Personalizes titles and summaries

### Frontend Changes (App.jsx)

```jsx
{/* Personalized Relevance Banner */}
{paper.patient_relevance && paper.is_personalized && (
  <div style={{ 
    background: 'linear-gradient(90deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15))', 
    padding: '0.625rem', 
    borderRadius: '0.5rem'
  }}>
    <div style={{ fontSize: '0.75rem', fontWeight: '600', color: '#15803d' }}>
      Personalized for Your Case
    </div>
    <div style={{ fontSize: '0.75rem', color: '#166534' }}>
      {paper.patient_relevance}
    </div>
  </div>
)}

{/* Personalized Badge */}
{paper.is_personalized && (
  <span style={{ fontSize: '0.75rem', color: '#22c55e' }}>
    <span>✨</span> Personalized
  </span>
)}

{/* Symptom Match Badge */}
{paper.symptom_relevance && (
  <span style={{ fontSize: '0.75rem', color: '#0891b2' }}>
    Symptom Match
  </span>
)}
```

## Example Scenarios

### Scenario 1: Elderly Patient with Heart Disease

**Patient Context:**
- Age: 72
- Gender: Male
- Symptoms: chest pain, shortness of breath, fatigue
- Condition: Coronary Artery Disease

**System Behavior:**
1. Searches for papers about Coronary Artery Disease
2. Prioritizes papers with keywords: "elderly", "geriatric", "older adults"
3. Boosts papers mentioning "chest pain", "shortness of breath", "fatigue"
4. Ranks papers by: relevance score (7.5) > year (2024) > significance (High)

**Result:**
```
★ Management of Coronary Artery Disease in Elderly Patients
✨ Personalized for Your Case
Discusses symptoms relevant to your case: chest pain, shortness of breath; 
Focuses on elderly patients, matching your age group

Authors: Smith J, et al. | Journal: Cardiology Today | Year: 2024
[✨ Personalized] [Symptom Match] [High Significance]
```

### Scenario 2: Pediatric Patient with Asthma

**Patient Context:**
- Age: 8
- Gender: Female
- Symptoms: wheezing, cough, difficulty breathing
- Condition: Asthma

**System Behavior:**
1. Searches for papers about Asthma
2. Prioritizes papers with keywords: "pediatric", "children", "adolescent"
3. Boosts papers mentioning "wheezing", "cough", "difficulty breathing"
4. Filters out adult-specific research

**Result:**
```
★ Pediatric Asthma Management: Current Guidelines and Treatment
✨ Personalized for Your Case
Discusses symptoms relevant to your case: wheezing, cough; 
Focuses on pediatric population, matching your age group; 
Includes gender-specific research relevant to female patients

Authors: Chen L, et al. | Journal: Pediatric Respiratory Medicine | Year: 2024
[✨ Personalized] [Symptom Match] [High Significance]
```

### Scenario 3: Adult with Multiple Symptoms

**Patient Context:**
- Age: 45
- Gender: Female
- Symptoms: fatigue, joint pain, rash, fever
- Condition: Lupus

**System Behavior:**
1. Searches for papers about Lupus
2. Matches adult age group
3. Boosts papers discussing multiple symptoms: fatigue, joint pain, rash, fever
4. Considers gender-specific aspects (lupus more common in women)

**Result:**
```
★ Systemic Lupus Erythematosus in Women: Multi-Symptom Presentation
✨ Personalized for Your Case
Discusses symptoms relevant to your case: fatigue, joint pain, rash; 
Includes gender-specific research relevant to female patients

Authors: Garcia M, et al. | Journal: Rheumatology International | Year: 2024
[✨ Personalized] [Symptom Match] [High Significance]
```

## Benefits

1. **Improved Relevance**: Patients see research most applicable to their specific case
2. **Better Decision Making**: Clinicians can reference age/symptom-appropriate studies
3. **Transparency**: Clear explanations of why each paper is recommended
4. **Evidence-Based**: Uses actual PubMed data with semantic understanding
5. **Real-Time Adaptation**: Recommendations change based on patient presentation
6. **Time Savings**: No need to manually filter through irrelevant research

## Performance Considerations

- **PubMedBERT Enhancement**: Only triggered for paper sets > 3 papers to optimize performance
- **Symptom Limit**: Checks top 10 symptoms to avoid excessive processing
- **Paper Limit**: Returns top 3-5 most relevant papers per condition
- **Caching**: Paper retrieval is efficient with in-memory processing
- **Fallback**: If PubMedBERT fails, system falls back to keyword matching

## Future Enhancements

1. **Lab Value Integration**: Factor in specific lab results for matching
2. **Comorbidity Matching**: Consider multiple conditions simultaneously
3. **Treatment Response**: Prioritize papers about treatment outcomes similar to patient
4. **Geographic Relevance**: Consider regional disease variants and guidelines
5. **Real-Time PubMed API**: Query live PubMed database for latest research
6. **User Feedback**: Allow clinicians to rate paper relevance for ML improvement

## Testing

To test the personalized research paper system:

```bash
# 1. Start the backend
cd cdss_chatbot
python manage.py runserver

# 2. Start the frontend
cd cdss-react-frontend
npm start

# 3. Test with different patient profiles:

# Test elderly patient
Query: "72 year old male with chest pain and shortness of breath"

# Test pediatric patient
Query: "8 year old child with wheezing and cough"

# Test adult with specific symptoms
Query: "45 year old female with fatigue, joint pain, and rash"
```

Check the research papers section for:
- ✨ Personalized badges
- Personalized relevance banners
- Symptom Match indicators
- Age-appropriate papers
- Relevant symptom mentions

## Conclusion

The system now provides **dynamic, real-time, patient-specific research paper recommendations** that adapt to each patient's unique presentation. This represents a significant improvement over templated information, ensuring clinicians and patients receive the most relevant evidence-based research for their specific case.

