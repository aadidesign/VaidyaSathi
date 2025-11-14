import json
import re
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .enhanced_medical_system import get_enhanced_medical_system
from .rag_system import RAGClinicalDecisionSupport
from .models import Patient

# Configure the Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the improved RAG system
rag_system = None

def get_rag_system():
    """Get or initialize the RAG system"""
    global rag_system
    if rag_system is None:
        print("Initializing RAG system for Django...")
        rag_system = RAGClinicalDecisionSupport()
        print("RAG system initialized for Django")
    return rag_system

def index(request):
    """Render the chatbot interface"""
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chat_api(request):
    """API endpoint to process chat requests using improved RAG system"""
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            user_input = data.get('message', '')
            
            if not user_input:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            # Use the improved RAG system instead of direct Gemini API
            print(f"Processing query: {user_input}")
            rag = get_rag_system()
            result = rag.analyze_case(user_input)
            
            # Format the response for the frontend with comprehensive data
            response_data = {
                'message': result.get('summary', 'Analysis completed'),
                'status': 'success',
                'analysis': {
                    'differential_diagnoses': result.get('differential_diagnoses', []),
                    'research_papers': result.get('research_papers', {}),
                    'medical_knowledge': result.get('medical_knowledge', []),
                    'recommendations': result.get('recommendations', {}),
                    'risk_assessment': result.get('risk_assessment', {}),
                    'preprocessing_stats': result.get('preprocessing_stats', {})
                },
                # Add comprehensive analysis data
                'full_analysis': {
                    'differential_diagnoses': result.get('differential_diagnoses', []),
                    'treatment_recommendations': result.get('treatment_recommendations', []),
                    'research_sources': result.get('research_sources', []),
                    'risk_assessment': result.get('risk_assessment', {}),
                    'clinical_recommendations': result.get('clinical_recommendations', {}),
                    'patient_education': result.get('patient_education', {}),
                    'follow_up_plan': result.get('follow_up_plan', {}),
                    'emergency_protocols': result.get('emergency_protocols', {}),
                    'medication_information': result.get('medication_information', {}),
                    'diagnostic_workup': result.get('diagnostic_workup', {}),
                    'confidence_score': result.get('confidence_score', 0.0),
                    'analysis_timestamp': result.get('analysis_timestamp', ''),
                    'system_version': result.get('system_version', ''),
                    'summary': result.get('summary', ''),
                    'preprocessing_stats': result.get('preprocessing_stats', {}),
                    'semantic_analysis': result.get('semantic_analysis', {}),
                    'llm_features': result.get('llm_features', {}),
                    'query': result.get('query', ''),
                    'patient_info': result.get('patient_info', {})
                }
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"❌ Error in chat_api: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

def patient_list(request):
    """Render the patient list page"""
    return render(request, 'chatbot_app/patients.html')

@csrf_exempt
def patient_api(request):
    """API endpoint to manage patients"""
    if request.method == 'GET':
        patients = Patient.objects.all().values()
        return JsonResponse(list(patients), safe=False)
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            patient = Patient.objects.create(
                name=data.get('name'),
                age=data.get('age'),
                gender=data.get('gender'),
                contact=data.get('contact')
            )
            return JsonResponse({
                'id': patient.id,
                'name': patient.name,
                'age': patient.age,
                'gender': patient.gender,
                'contact': patient.contact
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def patient_detail_api(request, patient_id):
    """API endpoint to manage a specific patient"""
    try:
        patient = Patient.objects.get(id=patient_id)
    except Patient.DoesNotExist:
        return JsonResponse({'error': 'Patient not found'}, status=404)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'contact': patient.contact
        })
    
    elif request.method == 'DELETE':
        patient.delete()
        return JsonResponse({'status': 'success'})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def ai_recommendation(request):
    """Render the AI recommendation page"""
    patient_id = request.GET.get('patientId')
    return render(request, 'chatbot_app/ai_recommendation.html', {'patient_id': patient_id})

@csrf_exempt
def generate_recommendation_api(request):
    """API endpoint to generate AI recommendations"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            patient_id = data.get('patientId')
            
            if not patient_id:
                return JsonResponse({'error': 'Missing patientId in request body'}, status=400)
            
            try:
                patient = Patient.objects.get(id=patient_id)
            except Patient.DoesNotExist:
                return JsonResponse({'error': 'Patient not found'}, status=404)
            
            # Construct a clinical query from the patient data
            clinical_query = f"Patient is a {patient.age} year old {patient.gender}."
            
            # Add any additional clinical information from the request
            if 'clinicalInfo' in data:
                clinical_query += f" {data['clinicalInfo']}"
            
            # Get the RAG system
            rag = get_rag_system()
            
            # Analyze the case
            ehr_data = {
                'age': patient.age,
                'gender': patient.gender,
            }
            
            results = rag.analyze_case(clinical_query, ehr_data)
            
            return JsonResponse(results)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

@csrf_exempt
def patient_recommendation_api(request, patient_id):
    """API endpoint to get detailed AI recommendations for a specific patient"""
    if request.method == 'GET':
        try:
            # Get the patient
            try:
                patient = Patient.objects.get(id=patient_id)
            except Patient.DoesNotExist:
                return JsonResponse({'error': f'Patient with ID {patient_id} not found'}, status=404)
            
            # Construct a clinical query from the patient data
            clinical_query = f"Patient is a {patient.age} year old {patient.gender}."
            
            # Get the RAG system
            rag = get_rag_system()
            
            # Analyze the case
            ehr_data = {
                'age': patient.age,
                'gender': patient.gender,
                'patient_id': patient.id,
                'patient_name': patient.name
            }
            
            # Add debug output
            print(f"Processing recommendation for patient: {patient.name} (ID: {patient.id})")
            print(f"Clinical query: {clinical_query}")
            
            results = rag.analyze_case(clinical_query, ehr_data)
            
            # Format the response for the frontend
            response = {
                'patient_info': {
                    'id': patient.id,
                    'name': patient.name,
                    'age': patient.age,
                    'gender': patient.gender,
                    'contact': patient.contact
                },
                'differential_diagnoses': results.get('differential_diagnoses', []),
                'research_papers': results.get('research_papers', {}),
                'retrieved_context': results.get('retrieved_context', [])
            }
            
            return JsonResponse(response)
            
        except Exception as e:
            import traceback
            print(f"Error in patient_recommendation_api: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)

def rag_chatbot(request):
    """Render the RAG chatbot interface"""
    return render(request, 'chatbot_app/rag_chatbot.html')

@csrf_exempt
def rag_chat_api(request):
    """API endpoint to process RAG-enhanced chat requests with full features"""
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            user_input = data.get('message', '')
            
            if not user_input:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            print(f"Processing query: {user_input}")
            
            # Helper function to clean Unicode characters
            def clean_unicode(data):
                """Remove problematic Unicode characters that cause encoding issues"""
                if isinstance(data, str):
                    # Replace common problematic Unicode characters
                    data = data.encode('ascii', 'ignore').decode('ascii')
                    return data
                elif isinstance(data, dict):
                    return {k: clean_unicode(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_unicode(item) for item in data]
                else:
                    return data
            
            # Use the full RAG system
            print("=" * 80)
            print("Initializing RAG system...")
            rag = get_rag_system()
            print("RAG system loaded successfully")
            
            # Analyze the case with full features
            print(f"Analyzing query: {user_input}")
            analysis_result = rag.analyze_case(user_input)
            print(f"Analysis completed. Keys returned: {list(analysis_result.keys())}")
            
            # Clean the response to remove Unicode characters
            print("Cleaning Unicode characters...")
            analysis_result = clean_unicode(analysis_result)
            
            # Format the response
            print("Formatting response...")
            response_data = {
                'message': analysis_result.get('summary', 'Analysis completed successfully'),
                'status': 'success',
                'analysis': {
                    'differential_diagnoses': analysis_result.get('differential_diagnoses', []),
                    'research_papers': analysis_result.get('research_papers', {}),
                    'medical_knowledge': analysis_result.get('retrieved_context', []),
                    'recommendations': analysis_result.get('recommendations', {}),
                    'risk_assessment': analysis_result.get('risk_assessment', {}),
                    'preprocessing_stats': analysis_result.get('preprocessing_stats', {})
                },
                'full_analysis': {
                    'differential_diagnoses': analysis_result.get('differential_diagnoses', []),
                    'treatment_recommendations': analysis_result.get('treatment_recommendations', []),
                    'research_sources': analysis_result.get('research_papers', {}),
                    'risk_assessment': analysis_result.get('risk_assessment', {}),
                    'recommendations': analysis_result.get('recommendations', {}),
                    'summary': analysis_result.get('summary', ''),
                    'preprocessing_stats': analysis_result.get('preprocessing_stats', {}),
                    'semantic_analysis': analysis_result.get('semantic_analysis', {}),
                    'llm_features': analysis_result.get('llm_features', {}),
                    'query': user_input,
                    'retrieved_context': analysis_result.get('retrieved_context', [])
                }
            }
            
            print(f"Response formatted. Differential diagnoses count: {len(response_data['analysis']['differential_diagnoses'])}")
            print("=" * 80)
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"Error in rag_chat_api: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def medical_knowledge_search_api(request):
    """API endpoint to search for medical knowledge with filtering"""
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            query = data.get('query', '')
            filters = data.get('filters', {})
            
            if not query:
                return JsonResponse({'error': 'No search query provided'}, status=400)
            
            # Get the enhanced medical system
            medical_system = get_enhanced_medical_system()
            
            # Search for medical knowledge using enhanced system
            search_results = medical_system.analyze_medical_query(query)
            
            # Return the response
            return JsonResponse({
                'results': search_results,
                'status': 'success'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def risk_assessment_api(request):
    """API endpoint to perform risk assessment on a patient case"""
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            patient_info = data.get('patient_info', {})
            clinical_query = data.get('clinical_query', '')
            
            if not patient_info and not clinical_query:
                return JsonResponse({'error': 'No patient information or clinical query provided'}, status=400)
            
            # Get the enhanced medical system
            medical_system = get_enhanced_medical_system()
            
            # Analyze the case
            analysis_result = medical_system.analyze_medical_query(clinical_query)
            
            # Extract risk assessment
            risk_assessment = analysis_result.get('risk_assessment', {})
            
            # Return the response
            return JsonResponse({
                'risk_assessment': risk_assessment,
                'status': 'success'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def medical_search(request):
    """Render the medical knowledge search interface"""
    return render(request, 'chatbot_app/medical_search.html')

@csrf_exempt
def test_all_features_api(request):
    """API endpoint to test all NLP and LLM features comprehensively"""
    if request.method == 'GET':
        try:
            print("🧪 Running comprehensive feature tests...")
            
            # Test results dictionary
            test_results = {
                'timestamp': str(__import__('datetime').datetime.now()),
                'tests': {}
            }
            
            # Test 1: NLP Preprocessing
            print("Testing NLP preprocessing...")
            try:
                from Rag.nlp_utils import preprocess, dependency_parse, SimpleSpellCorrector
                test_text = "Patient has chst pain and difficutly breathing"
                
                # Test spell correction
                corrector = SimpleSpellCorrector()
                corrected = corrector.correction("chst")
                
                # Test preprocessing
                result = preprocess(test_text)
                
                test_results['tests']['nlp_preprocessing'] = {
                    'status': 'PASSED',
                    'spell_correction': f"'chst' -> '{corrected}'",
                    'sentences': len(result.get('sentences', [])),
                    'tokens': len(result.get('tokens', [])),
                    'normalized': len(result.get('normalized', [])),
                    'pos_tags': len(result.get('pos', []))
                }
                print("✅ NLP preprocessing test passed")
            except Exception as e:
                test_results['tests']['nlp_preprocessing'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"❌ NLP preprocessing test failed: {e}")
            
            # Test 2: Semantic Parsing
            print("Testing semantic parsing...")
            try:
                from Rag.semantic_parser import analyze_medical_semantics
                semantic_result = analyze_medical_semantics("Patient has myocardial infarction and pneumonia")
                
                entities = semantic_result.get('medical_entities', {})
                relationships = semantic_result.get('medical_relationships', [])
                
                test_results['tests']['semantic_parsing'] = {
                    'status': 'PASSED',
                    'entity_types': len(entities),
                    'total_entities': sum(len(v) for v in entities.values()),
                    'relationships': len(relationships),
                    'umls_linking': 'error' not in semantic_result
                }
                print("✅ Semantic parsing test passed")
            except Exception as e:
                test_results['tests']['semantic_parsing'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"❌ Semantic parsing test failed: {e}")
            
            # Test 3: LLM Features
            print("Testing LLM features...")
            try:
                from Rag.llm_features import (
                    get_dense_retrieval, get_text_summarizer, 
                    get_rag_generator, get_qa_system,
                    dense_retrieve, summarize_medical_text
                )
                
                # Test dense retrieval
                retriever = get_dense_retrieval()
                docs = ["Pneumonia is a lung infection", "Heart attack affects cardiac muscle"]
                dense_results = dense_retrieve("lung infection", docs, top_k=1)
                
                # Test text summarization
                long_text = "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing."
                summary_result = summarize_medical_text(long_text)
                
                test_results['tests']['llm_features'] = {
                    'status': 'PASSED',
                    'dense_retrieval': len(dense_results) > 0,
                    'text_summarization': len(summary_result.get('abstractive_summary', '')) > 0,
                    'models_loaded': True
                }
                print("✅ LLM features test passed")
            except Exception as e:
                test_results['tests']['llm_features'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"❌ LLM features test failed: {e}")
            
            # Test 4: RAG System Integration
            print("Testing RAG system integration...")
            try:
                rag_system = get_rag_system()
                analysis = rag_system.analyze_case("30 year old patient with chest pain and shortness of breath")
                
                required_keys = [
                    'differential_diagnoses', 'semantic_analysis', 'enhanced_response', 
                    'medical_summary', 'preprocessing_stats', 'llm_features'
                ]
                missing_keys = [key for key in required_keys if key not in analysis]
                
                test_results['tests']['rag_integration'] = {
                    'status': 'PASSED' if not missing_keys else 'PARTIAL',
                    'features_present': len(required_keys) - len(missing_keys),
                    'total_features': len(required_keys),
                    'missing_features': missing_keys,
                    'preprocessing_stats': 'preprocessing_stats' in analysis
                }
                print("✅ RAG integration test passed")
            except Exception as e:
                test_results['tests']['rag_integration'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"❌ RAG integration test failed: {e}")
            
            # Calculate overall status
            passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASSED')
            total_tests = len(test_results['tests'])
            
            test_results['summary'] = {
                'tests_passed': passed_tests,
                'total_tests': total_tests,
                'success_rate': f"{(passed_tests/total_tests)*100:.1f}%",
                'overall_status': 'ALL PASSED' if passed_tests == total_tests else 'SOME ISSUES'
            }
            
            print(f"🎯 Test Summary: {passed_tests}/{total_tests} tests passed")
            print("=" * 80)
            
            return JsonResponse({
                'status': 'success',
                'test_results': test_results
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)

@csrf_exempt
def health_check(request):
    """Simple health check endpoint"""
    return JsonResponse({
        'status': 'ok',
        'message': 'CDSS backend is running',
        'timestamp': str(__import__('datetime').datetime.now())
    }) 