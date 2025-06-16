import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk
import os
import tempfile
import re

# --- GESTION DE NLTK PUNKT POUR SENT_TOKENIZE ---
nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

from nltk.tokenize import sent_tokenize

# --- CONFIG PAGE STREAMLIT ---
st.set_page_config(page_title="🤖 Chatbot Histoire Côte d'Ivoire", layout="wide")

st.markdown("""
    <style>
        .chat-container {
            border: 1px solid #CCC;
            border-radius: 10px;
            padding: 20px;
            background-color: #F9F9F9;
            max-width: 700px;
            margin: auto;
        }
        .bot-message {
            background-color: #E6F0FA;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            white-space: pre-wrap;
        }
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DU MODELE ---
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- CHARGEMENT DU TEXTE ---
def load_text():
    try:
        with open("Histoire.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error("❌ Fichier 'Histoire.txt' introuvable. Veuillez vérifier son emplacement.")
        st.stop()

text = load_text()

# --- TOKENISATION ET EMBEDDINGS ---
sentences = sent_tokenize(text)
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# --- FONCTIONS UTILES ---
def preprocess_input(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # enlever ponctuation
    return text

def get_theme_based_suggestions(user_input):
    themes = {
        "histoire": [
            "Quand la Côte d'Ivoire a-t-elle obtenu son indépendance ?",
            "Qui était Félix Houphouët-Boigny ?",
            "Quels royaumes existaient avant la colonisation ?"
        ],
        "géographie": [
            "Quelle est la superficie de la Côte d'Ivoire ?",
            "Quels sont les fleuves principaux du pays ?",
            "Quel est le climat dans le nord du pays ?"
        ],
        "politique": [
            "Quel est le système politique de la Côte d'Ivoire ?",
            "Comment est organisée l'administration territoriale ?"
        ],
        "culture": [
            "Quels sont les plats traditionnels ivoiriens ?",
            "Quelles langues sont parlées en Côte d'Ivoire ?",
            "Quels styles musicaux sont populaires ?"
        ],
        "sport": [
            "Combien de fois la Côte d'Ivoire a-t-elle gagné la CAN ?",
            "Quels sports sont populaires en Côte d'Ivoire ?"
        ],
        "économie": [
            "Quel est le rôle du cacao dans l'économie ?",
            "Quels secteurs économiques sont en croissance ?"
        ]
    }

    for theme, questions in themes.items():
        if theme in user_input:
            return questions

    return [
        "Quelle est la population de la Côte d'Ivoire ?",
        "Quels sont les atouts touristiques du pays ?",
        "Quels sont les enjeux actuels du pays ?"
    ]

def get_most_relevant_sentence(user_input, base_threshold=0.4):
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_result = int(similarities.argmax())
    best_score = float(similarities[top_result])

    length = len(user_input.split())
    threshold = max(0.3, min(0.6, base_threshold + 0.01 * (5 - length)))

    if best_score < threshold:
        suggestions = get_theme_based_suggestions(user_input)
        return (
            "🤖 Je suis désolé, je n'ai pas trouvé de réponse claire à votre question.\n\n"
            "Voici quelques suggestions que vous pouvez essayer :\n\n" +
            "\n".join(f"- **{q}**" for q in suggestions)
        )

    response = sentences[top_result]
    if top_result + 1 < len(sentences):
        response += " " + sentences[top_result + 1]
    return response

def chatbot(user_input):
    user_input_clean = preprocess_input(user_input)
    greetings = ["bonjour", "salut", "hello", "coucou"]
    if user_input_clean in greetings:
        return "Bonjour ! Je suis votre assistant sur la Côte d'Ivoire. Posez-moi une question !"
    return get_most_relevant_sentence(user_input_clean)

# --- INTERFACE PRINCIPALE ---
def main():
    st.title("Chatbot interactif sur l'histoire de la Côte d'Ivoire")
    st.markdown("Posez une question sur l'histoire, la politique, la culture, l'économie ou le sport en Côte d'Ivoire.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Votre question :")
        submitted = st.form_submit_button("Envoyer")

    if submitted and user_input:
        response = chatbot(user_input)
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", response))

    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for sender, msg in st.session_state.messages:
            if sender == "user":
                st.markdown(f'<div class="user-message">🧑 {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {msg}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
