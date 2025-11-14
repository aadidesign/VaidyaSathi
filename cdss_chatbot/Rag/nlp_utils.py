import re
from typing import List, Dict, Tuple

try:
    import spacy  # Optional, used if available for robust lemmatization
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None
    spacy = None

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, data as nltk_data
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


# Ensure resources (nltk downloads done in rag_system as well; keep here for CLI usage)
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Ensure POS tagger (handle both new and legacy resource names)
def _ensure_pos_tagger():
    try:
        nltk_data.find('taggers/averaged_perceptron_tagger_eng')
        return 'averaged_perceptron_tagger_eng'
    except Exception:
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk_data.find('taggers/averaged_perceptron_tagger_eng')
            return 'averaged_perceptron_tagger_eng'
        except Exception:
            try:
                nltk_data.find('taggers/averaged_perceptron_tagger')
                return 'averaged_perceptron_tagger'
            except Exception:
                try:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk_data.find('taggers/averaged_perceptron_tagger')
                    return 'averaged_perceptron_tagger'
                except Exception:
                    return None

_POS_TAGGER = _ensure_pos_tagger()


_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop = set(stopwords.words("english"))


def sentence_segment(text: str) -> List[str]:
    """Robust sentence segmentation using NLTK."""
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        # Fallback split on punctuation
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _simple_medical_compound_split(token: str) -> List[str]:
    """
    Custom word segmentation for common medical compounds and camelCase/kebab_case/underscore.
    Examples: "COVID19Positive" -> ["covid19", "positive"]
              "post-op_pain" -> ["post", "op", "pain"]
    """
    # Split on underscores, hyphens, slashes
    parts = re.split(r"[\-_/]+", token)
    new_parts = []
    for p in parts:
        # Split camelCase or PascalCase
        split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
        new_parts.extend(split_camel.split())
    # Lowercase and filter
    return [p.lower() for p in new_parts if p]


def word_segment(text: str) -> List[str]:
    """Word tokenization with custom compound splitting and normalization."""
    tokens = []
    for tok in word_tokenize(text):
        if re.search(r"[\-_/A-Z]", tok):
            tokens.extend(_simple_medical_compound_split(tok))
        else:
            tokens.append(tok.lower())
    return tokens


def lemmatize(tokens: List[str]) -> List[str]:
    if _SPACY_NLP:
        doc = _SPACY_NLP(" ".join(tokens))
        return [t.lemma_.lower() for t in doc if t.lemma_.strip()]
    # NLTK fallback
    result = []
    for t in tokens:
        if not t.isalpha():
            continue
        if t in _stop:
            continue
        # crude POS heuristic
        pos = 'n'
        if t.endswith('ing') or t.endswith('ed'):
            pos = 'v'
        result.append(_lemmatizer.lemmatize(t, pos))
    return result


def stem(tokens: List[str]) -> List[str]:
    return [_stemmer.stem(t) for t in tokens]


def finite_state_validator(tokens: List[str]) -> List[str]:
    """
    Very small FSM to remove tokens that are just punctuation or repeated noise.
    States: START -> ALNUM | PUNCT; if PUNCT and token non-meaningful -> drop
    """
    cleaned = []
    for t in tokens:
        if re.fullmatch(r"[\W_]+", t):
            # PUNCT state -> drop
            continue
        cleaned.append(t)
    return cleaned


def preprocess(text: str, *, use_stemming: bool = False) -> Dict[str, List[str]]:
    """Full pipeline: sentence segmentation, word segmentation, POS tagging, lemmatization/stemming, FSM cleanup."""
    sentences = sentence_segment(text)
    tokens = [tok for s in sentences for tok in word_segment(s)]
    tokens = finite_state_validator(tokens)
    # POS tagging if tagger available; otherwise, skip gracefully
    try:
        tagged = pos_tag(tokens) if _POS_TAGGER else []
    except Exception:
        tagged = []
    norm = stem(tokens) if use_stemming else lemmatize(tokens)
    return {
        "sentences": sentences,
        "tokens": tokens,
        "pos": tagged,
        "normalized": norm,
    }


# Spell correction (Norvig-style) with optional medical dictionary bias
class SimpleSpellCorrector:
    def __init__(self, vocabulary: List[str] = None):
        self.N = {}
        if vocabulary:
            for w in vocabulary:
                w = w.lower()
                self.N[w] = self.N.get(w, 0) + 1

    def known(self, words):
        return {w for w in words if w in self.N}

    def edits1(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def candidates(self, word):
        return (self.known([word]) or
                self.known(self.edits1(word)) or
                [word])

    def correction(self, word):
        candidates = list(self.candidates(word.lower()))
        candidates.sort(key=lambda w: self.N.get(w, 1), reverse=True)
        return candidates[0] if candidates else word


def dependency_parse(text: str):
    """Return spaCy dependency parse if model available, else empty list."""
    if _SPACY_NLP is None or spacy is None:
        return []
    doc = _SPACY_NLP(text)
    return [(t.text, t.lemma_, t.pos_, t.dep_, t.head.text) for t in doc]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


