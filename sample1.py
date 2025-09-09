import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
import logging
from nltk.corpus import stopwords
from transformers import pipeline

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Abstract (Display in Streamlit for context) ---
abstract_text = """
Leveraging Neural Machine Translation and Annotation Projection for Developing Multilingual Named Entity Recognition in Clinical NLP

**Keywords:** Neural Machine Translation, annotation projection, clinical NLP, Named Entity Recognition, multilingual datasets, Catalan, expert validation.
"""

# --- Expanded English Clinical Terms ---
clinical_terms_english = [
    "fever", "cold", "cough", "headache", "vomiting", "diarrhea", "abdominal_pain", "fatigue", "weakness", "dizziness",
    "skin rash", "itching", "burning_sensation", "joint pain", "muscle pain", "swelling", "shortness of breath",
    "wheezing", "chest pain", "palpitation", "nausea", "constipation", "loss of appetite", "weight loss", "weight gain",
    "blurred vision", "double vision", "eye pain", "ear pain", "hearing loss", "sore throat", "runny nose", "sneezing",
    "body ache", "chill", "sweating", "insomnia", "anxiety", "depression", "irritability", "memory loss", "confusion",
    "seizure", "paralysis", "numbness", "tingling", "back pain", "neck pain", "knee pain", "leg pain", "foot pain",
    "arm pain", "hand pain", "shoulder pain", "hip pain", "redness", "warmth", "pus", "discharge", "bleeding",
    "bruising", "jaundice", "dark urine", "pale stool", "bloody stool", "blood in urine", "high_blood_pressure",
    "low blood pressure", "sugar", "diabetes", "malaria", "dengue", "typhoid", "cholera", "tuberculosis", "leprosy",
    "filariasis", "chickenpox", "measles", "mumps", "rubella", "conjunctivitis", "scabies", "ringworm", "heat stroke",
    "sunburn", "snake bite", "rabies", "heartburn", "indigestion", "acidity", "bloating", "flatulence", "constipation",
    "piles", "appendicitis", "hernia", "ulcer", "gallstone", "stomach ache",
    "pain", "infection", "inflammation", "allergy", "asthma", "bronchitis", "pneumonia", "flu", "virus", "bacteria",
    "fungus", "cancer", "tumor", "arthritis", "osteoporosis", "heart disease", "stroke", "anemia", "thyroid",
    "migraine", "rash", "wound", "cut", "fracture", "sprain", "strain", "burn", "cyst", "polyp", "lesion", "abscess",
    "edema", "ischemia", "necrosis", "hemorrhage", "thrombosis", "embolism", "fibrosis", "stenosis", "dilation",
    "hypertrophy", "atrophy", "metaplasia", "dysplasia", "neoplasia", "carcinoma", "sarcoma", "lymphoma", "leukemia",
    "melanoma", "adenoma", "papilloma", "myoma", "lipoma", "neuroma", "glioma", "meningioma", "blastoma", "cystitis",
    "urethritis", "pyelonephritis", "glomerulonephritis", "hepatitis", "cirrhosis", "pancreatitis", "cholecystitis",
    "gastritis", "enteritis", "colitis", "diverticulitis", "appendicitis", "peritonitis", "meningitis", "encephalitis",
    "myelitis", "neuritis", "myositis", "arthritis", "bursitis", "tendonitis", "osteomyelitis", "sepsis", "shock",
    "trauma", "injury", "poisoning", "overdose", "withdrawal", "dependence", "addiction", "tolerance", "resistance",
    "mutation", "gene", "chromosome", "dna", "rna", "protein", "enzyme", "hormone", "receptor", "antibody", "antigen",
    "vaccine", "therapy", "treatment", "diagnosis", "prognosis", "syndrome", "disorder", "condition", "disease"
]
clinical_terms_english = [term.lower() for term in clinical_terms_english]  # Convert to lowercase

# --- Load NMT Models ---
@st.cache_resource
def load_nmt_models():
    """Loads NMT models for translation."""
    logging.info("Loading NMT models...")
    translator_es = pipeline('translation', model='Helsinki-NLP/opus-mt-en-es')  # English to Spanish
    translator_es_ca = pipeline('translation', model='Helsinki-NLP/opus-mt-es-ca')  # Spanish to Catalan
    logging.info("NMT models loaded.")
    return translator_es, translator_es_ca

translator_es, translator_es_ca = load_nmt_models()  # Load models at startup

# --- Functions ---
def replace_multi_word_terms(text, replacements):
    """Replaces multi-word terms with single tokens using regular expressions."""
    for phrase, token in replacements.items():
        pattern = r'\b' + re.escape(phrase) + r'\b'  # Word boundary matching
        text = re.sub(pattern, token, text)
    return text

def find_closest_match(term, term_list, threshold=70, use_partial=True):
    """Finds the closest match in a list of terms using fuzzy matching."""
    best_match = None
    best_score = 0
    for candidate in term_list:
        score1 = fuzz.ratio(term, candidate)
        score2 = fuzz.partial_ratio(term, candidate) if use_partial else 0
        score = max(score1, score2)

        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    return best_match

def translate_terms_nmt(terms, translator_es, translator_es_ca):
    """Translates a list of terms using NMT models (English -> Spanish -> Catalan)."""
    spanish_translations = []
    catalan_translations = []

    for term in terms:
        try:
            # English to Spanish
            spanish_result = translator_es(term)
            spanish_translation = spanish_result[0]['translation_text']
            spanish_translations.append(spanish_translation)

            # Spanish to Catalan
            catalan_result = translator_es_ca(spanish_translation)
            catalan_translation = catalan_result[0]['translation_text']
            catalan_translations.append(catalan_translation)

        except Exception as e:
            logging.error(f"NMT translation error for term '{term}': {e}")
            spanish_translations.append(f"Translation Error: {term}")
            catalan_translations.append(f"Translation Error: {term}") # Make sure to still add placeholder for Catalan

    return spanish_translations, catalan_translations

def identify_and_translate(sentence, clinical_terms, lemmatizer, fuzzy_threshold=70, translator_es=None, translator_es_ca=None):
    """
    Identifies clinical terms (NER - in a basic way here using list matching), then translates them to Spanish and Catalan using NMT.
    """
    sentence = sentence.lower()

    # Preprocess: Replace multi-word terms with single tokens
    replacements = {
        "abdominal pain": "abdominal_pain",
        "burning sensation": "burning_sensation",
        "high blood pressure": "high_blood_pressure",
        "stomach ache": "abdominal_pain"
    }
    sentence = replace_multi_word_terms(sentence, replacements)

    tokens = word_tokenize(sentence)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    identified_terms = []
    annotations = []  # Keep track of which tokens are identified as terms
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        closest_match = find_closest_match(lemma, clinical_terms, threshold=fuzzy_threshold)
        if closest_match:
            identified_terms.append(closest_match)
            annotations.append(1)  # 1 indicates that the term was identified as a clinical term
        else:
            annotations.append(0)  # 0 indicates it's not a clinical term

    # Use NMT for translation
    if translator_es and translator_es_ca:
        spanish_translations, catalan_translations = translate_terms_nmt(identified_terms, translator_es, translator_es_ca)
    else:
        spanish_translations = ["NMT Model Error"] * len(identified_terms)
        catalan_translations = ["NMT Model Error"] * len(identified_terms)
        logging.error("NMT translators not loaded properly.")

    return {
        "tokens": tokens, #return the tokens
        "identified_terms": identified_terms,
        "spanish_translations": spanish_translations,
        "catalan_translations": catalan_translations,
        "annotations": annotations,  # Return the annotations
    }

def evaluate_accuracy(sentence, clinical_terms, lemmatizer, true_labels, fuzzy_threshold=70):
    """
    Evaluates the accuracy of term identification.
    """
    sentence = sentence.lower()
    replacements = {
        "abdominal pain": "abdominal_pain",
        "burning sensation": "burning_sensation",
        "high blood pressure": "high_blood_pressure",
        "stomach ache": "abdominal_pain"
    }
    sentence = replace_multi_word_terms(sentence, replacements)

    tokens = word_tokenize(sentence)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Debug: Print tokens and true_labels
    # st.write("Tokens after preprocessing:", tokens)
    # st.write("True labels:", true_labels)

    predicted_labels = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        closest_match = find_closest_match(lemma, clinical_terms, threshold=fuzzy_threshold)
        if closest_match:
            predicted_labels.append(True)
        else:
            predicted_labels.append(False)

    # Ensure the lists are the same length
    min_length = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_length]
    predicted_labels = predicted_labels[:min_length]

    # Debug: Print predicted labels
    st.write("Predicted labels:", predicted_labels)

    # return accuracy, precision, recall, f1

def main():
    st.title("Clinical Term Identification (NER) and Translation (NMT)")
    st.markdown(abstract_text)

    user_sentence = st.text_area("Enter an English sentence with potential clinical terms:", "I have a fever and diarrhea, also some abdominal pain and dizziness. I felt burning sensation. Additionally, I suffer from shortness of breath and occasional chest pain, a little stomach ache.")

    # Example true labels - adjust based on the sentence above
    example_true_labels = [False, True, True, True, True, False, True, True, False, True, True, False] #This is wrong. It has to have length of tokens array

    fuzzy_threshold = st.slider("Fuzzy Matching Threshold:", min_value=1, max_value=100, value=70, step=1)

    if st.button("Analyze"):
        lemmatizer = WordNetLemmatizer()
        results = identify_and_translate(user_sentence, clinical_terms_english, lemmatizer, fuzzy_threshold=fuzzy_threshold, translator_es=translator_es, translator_es_ca=translator_es_ca)

        st.subheader("Results:")

        if results['identified_terms']:
            st.write("Identified English Clinical Terms:")
            st.write(results['identified_terms'])

            # Print the sentence with annotations:
            st.write("Sentence with Annotations:")
            annotated_sentence = ""
            for i, token in enumerate(results['tokens']):
                if results['annotations'][i] == 1:
                    annotated_sentence += f"**{token}** "  # Bold clinical terms
                else:
                    annotated_sentence += token + " "
            st.markdown(annotated_sentence)

            st.write("Spanish NER:")
            st.write(results['spanish_translations'])

            st.write("Catalan NER:")
            st.write(results['catalan_translations'])




        else:
            st.info("No clinical terms identified in the input sentence with the current threshold.")

        # Evaluate accuracy
        with st.expander("Evaluation (Debug)"):
            # This is where you provide the 'ground truth' annotations.
            # It's crucial that the length of `true_labels` matches the number of *tokens* in your preprocessed sentence.

            # Properly annotated sentence: "I have a fever and diarrhea, also some abdominal pain and dizziness. I felt burning sensation. Additionally, I suffer from shortness of breath and occasional chest pain, a little stomach ache."
            # Tokenized: ['fever', 'diarrhea', 'abdominal_pain', 'dizziness', 'burning_sensation', 'shortness', 'breath', 'chest', 'pain', 'stomach_ache']

            # Define the correct labels for each token:
            true_labels = [True, True, True, True, True, True, True, True, True, True]  # Example: 1 for clinical term, 0 otherwise.

            lemmatizer = WordNetLemmatizer()
            evaluate_accuracy(user_sentence, clinical_terms_english, lemmatizer, true_labels, fuzzy_threshold)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    main()