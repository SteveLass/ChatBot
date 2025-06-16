import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

# Configuration de la page
st.set_page_config(page_title="🤖 Chatbot Histoire Côte d'Ivoire", layout="wide")

# Fonction simple pour splitter les phrases
def simple_sentence_split(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# Chargement du modèle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chargement du texte
with open("Histoire.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Découpage simple sans nltk
sentences = simple_sentence_split(text)
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Fonction de récupération de la réponse pertinente
def get_most_relevant_sentence(user_input, threshold=0.4):
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_result = int(similarities.argmax())
    best_score = float(similarities[top_result])

    if best_score < threshold:
        return "🤖 Désolé, je n'ai pas trouvé de réponse claire à votre question."

    response = sentences[top_result]
    if top_result + 1 < len(sentences):
        response += " " + sentences[top_result + 1]
    return response

# Chatbot simple
def chatbot(user_input):
    if user_input.lower() in ["bonjour", "salut", "hello"]:
        return "Bonjour ! Posez-moi une question sur la Côte d'Ivoire."
    return get_most_relevant_sentence(user_input)

# Interface principale
def main():
    st.title("Chatbot interactif sur l'histoire de la Côte d'Ivoire")
    st.markdown("Posez une question sur l'histoire, la politique, la culture, l'économie ou le sport.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Votre question :")
        submitted = st.form_submit_button("Envoyer")

    if submitted and user_input:
        response = chatbot(user_input)
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", response))

    for sender, msg in st.session_state.messages:
        if sender == "user":
            st.markdown(f'<div style="text-align:right; background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px;">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:#E6F0FA; padding:10px; border-radius:10px; margin:5px;">🤖 {msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
