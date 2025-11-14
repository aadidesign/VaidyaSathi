from django.db import models
from django.utils import timezone

class Patient(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    contact = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name

class ChatSession(models.Model):
    """Model to store chat sessions and their analysis results"""
    session_id = models.CharField(max_length=100, unique=True)
    user_query = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    
    # Analysis results
    differential_diagnoses = models.JSONField(default=list, blank=True)
    risk_assessment = models.JSONField(default=dict, blank=True)
    semantic_analysis = models.JSONField(default=dict, blank=True)
    enhanced_response = models.JSONField(default=dict, blank=True)
    medical_summary = models.JSONField(default=dict, blank=True)
    qa_results = models.JSONField(default=list, blank=True)
    llm_features_used = models.JSONField(default=dict, blank=True)
    
    # Metadata
    response_time_ms = models.IntegerField(default=0)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Chat {self.session_id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class MedicalEntity(models.Model):
    """Model to store extracted medical entities for analytics"""
    chat_session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='entities')
    entity_type = models.CharField(max_length=50)  # DISEASE, DRUG, SYMPTOM, etc.
    entity_text = models.CharField(max_length=200)
    umls_cui = models.CharField(max_length=20, blank=True, null=True)
    confidence_score = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"{self.entity_type}: {self.entity_text}"

class AnalyticsData(models.Model):
    """Model to store analytics data for the CDSS system"""
    date = models.DateField(default=timezone.now)
    total_queries = models.IntegerField(default=0)
    successful_analyses = models.IntegerField(default=0)
    avg_response_time_ms = models.FloatField(default=0.0)
    
    # Feature usage statistics
    semantic_analysis_used = models.IntegerField(default=0)
    dense_retrieval_used = models.IntegerField(default=0)
    text_summarization_used = models.IntegerField(default=0)
    qa_system_used = models.IntegerField(default=0)
    enhanced_rag_used = models.IntegerField(default=0)
    
    # Most common entities
    top_medical_entities = models.JSONField(default=list, blank=True)
    common_diagnoses = models.JSONField(default=list, blank=True)
    
    class Meta:
        unique_together = ['date']
        ordering = ['-date']
    
    def __str__(self):
        return f"Analytics for {self.date}" 