import os

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from GeminiUtility import (load_gemini_pro_model,
                           gemini_flash_vision_response,
                           Gemini_Embeddings_model,
                           gemini_flash_response)

working_dir = os.path.dirname(os.path.abspath(__file__))

#setting up the page configuration

st.set_page_config(
    page_title="Gemini Ai",
    page_icon="✨",
    layout="centered"
)

with st.sidebar:
    selected = option_menu("Gen AI",
                           ["Chat Bot",
                            "Image Captioning",
                            "Embed text",
                            "Ask me anything"],
                           menu_icon='robot',icons=['chat-dots-fill',
                                                    'image-fill',
                                                    'textarea-t',
                                                    'patch-question-fill'],
                           default_index=0)

#function role between gemini pro and streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role

if selected == "Chat Bot":

    model = load_gemini_pro_model()

    #Initialize chat session in streamlit if not already present

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])


    #streamlit page title
    st.title("🤖 Chat Bot")
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    #input field for user's message

    user_prompt = st.chat_input("Ask Gemini-3.5...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        #display gemini-pro response

        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

#Image caption page
if selected == "Image Captioning":
    st.title("📷 Image Captioning")

    upload_image = st.file_uploader("Upload a file..",type=['jpg , jpeg','png'])
    if st.button("Generate Caption"):
        image = Image.open(upload_image)

        col1,col2 = st.columns(2)

        with col1:
            resized_image = image.resize((800,500))
            st.image(resized_image)

        default_prompt = "write a short caption for this image"

        caption = gemini_flash_vision_response(default_prompt,image)

        with col2:
            st.info(caption)

#Embedding model

if selected == "Embed text":
    st.title("🔡 Embed Text into vectors")

    #text = st.text_area("Write something to get vectors?")

    input_text = st.text_area(label="",placeholder="Enter the text to get Embeddings")

    if st.button("Generate Embeddings"):
        Response_vector = Gemini_Embeddings_model(input_text)
        st.markdown(Response_vector)
        #st.info(Embed_model_vectors)

#get response from gemini model

if selected == "Ask me anything":

    st.title("🧠 Ask Gemini flash")

    user_prompt = st.text_area(label="",placeholder="Ask gemini 3.5 flash...")

    if st.button("Get Response"):
        response = gemini_flash_response(user_prompt)

        st.markdown(response)

