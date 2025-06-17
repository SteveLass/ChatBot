import streamlit as st
import speech_recognition as sr
import nltk
import random
import string
from nltk.chat.util import Chat, reflections

# Initialisation de nltk
nltk.download('punkt')

# Chargement et préparation du texte
with open("base_connaissance.txt", "r", encoding="utf-8") as file:
    raw_text = file.read().lower()

sent_tokens = nltk.sent_tokenize(raw_text)
word_tokens = nltk.word_tokenize(raw_text)

# Réponses par défaut
default_responses = [
    "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?",
    "Hmm... essayez une autre question.",
    "Je suis encore en apprentissage. Soyez plus précis."
]

# Fonction de réponse simple (à améliorer avec TF-IDF ou modèle préentraîné)
def chatbot_response(user_input):
    user_input = user_input.lower()
    for sentence in sent_tokens:
        if user_input in sentence:
            return sentence
    return random.choice(default_responses)

# Fonction de reconnaissance vocale
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parlez maintenant...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="fr-FR")
            st.success(f"Vous avez dit : {text}")
            return text
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas compris.")
            return ""
        except sr.RequestError:
            st.error("Service de reconnaissance vocale indisponible.")
            return ""

# Interface Streamlit
st.title("🤖 Chatbot Vocal et Textuel")

mode = st.radio("Choisissez le mode d'entrée :", ("Texte", "Voix"))

if mode == "Texte":
    user_input = st.text_input("Entrez votre question :")
    if st.button("Envoyer") and user_input:
        response = chatbot_response(user_input)
        st.markdown(f"**Chatbot :** {response}")

else:
    if st.button("🎙️ Parler"):
        user_input = speech_to_text()
        if user_input:
            response = chatbot_response(user_input)
            st.markdown(f"**Chatbot :** {response}")
