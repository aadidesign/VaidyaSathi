from django.apps import AppConfig

class ChatbotAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Rag'

    def ready(self):
        # Ensure NLTK POS tagger resource is present at startup; fail-soft
        try:
            import nltk
            from nltk import data as nltk_data
            try:
                nltk_data.find('taggers/averaged_perceptron_tagger_eng')
            except Exception:
                try:
                    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                except Exception:
                    pass
            # Legacy fallback
            try:
                nltk_data.find('taggers/averaged_perceptron_tagger')
            except Exception:
                try:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                except Exception:
                    pass
        except Exception:
            # Do not crash the app if downloads fail
            pass