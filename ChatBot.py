import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# Initialisation de NLTK
nltk.download('punkt')

# Configuration de la page
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
        }
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
        }
        .icon {
            font-weight: bold;
            margin-right: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Chargement du modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chargement du texte
with open("Histoire.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Traitement du texte
sentences = sent_tokenize(text)
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Suggestions thématiques
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

    user_input_lower = user_input.lower()
    for theme, questions in themes.items():
        if theme in user_input_lower:
            return questions

    return [
        "Quelle est la population de la Côte d'Ivoire ?",
        "Quels sont les atouts touristiques du pays ?",
        "Quels sont les enjeux actuels du pays ?"
    ]

# Récupération de la réponse la plus pertinente
def get_most_relevant_sentence(user_input, threshold=0.4):
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_result = int(similarities.argmax())
    best_score = float(similarities[top_result])

    if best_score < threshold:
        suggestions = get_theme_based_suggestions(user_input)
        return (
            "🤖 Je suis désolé, je n'ai pas trouvé de réponse claire à votre question.\n\n"
            "Voici quelques suggestions que vous pouvez essayer :\n\n" +
            "\n".join(f"• {q}" for q in suggestions)
        )

    response = sentences[top_result]
    if top_result + 1 < len(sentences):
        response += " " + sentences[top_result + 1]
    return response

# Fonction de chatbot
def chatbot(user_input):
    if user_input.lower() in ["bonjour", "salut", "hello"]:
        return "Bonjour ! Je suis votre assistant sur la Côte d'Ivoire. Posez-moi une question !"
    return get_most_relevant_sentence(user_input)

# Interface principale
def main():
    st.title("Chatbot interactif sur l'histoire de la Côte d'Ivoire")
    st.markdown("Posez une question sur l'histoire, la politique, la culture, l'économie ou le sport en Côte d'Ivoire.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("chat_form"):
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
