import json
import os

import warnings
warnings.filterwarnings('ignore')

import google.generativeai as genai

working_dir = os.path.dirname(os.path.abspath(__file__))
#set path to config file
config_file_path = f"{working_dir}/config.json"

#load api key from json config file
config_data = json.load(open(config_file_path))

#load into genai
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

#configure google.generativeai with api key

genai.configure(api_key=GOOGLE_API_KEY)

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-3.5-flash")
    return gemini_pro_model

def gemini_flash_vision_response(prompt,image):
    gemini_flash_vision_model = genai.GenerativeModel("gemini-3.5-flash")
    response = gemini_flash_vision_model.generate_content([prompt,image])
    result = response.text
    return result

def Gemini_Embeddings_model(input_text):
    embedding_model = "gemini-embedding-2"
    embedding = genai.embed_content(model=embedding_model,content=input_text,task_type="retrieval_document")
    return embedding['embedding']

#get response from gemini 3.5

def gemini_flash_response(user_prompt):
    gemini_flash_model = genai.GenerativeModel("gemini-3.5-flash")
    response = gemini_flash_model.generate_content(user_prompt)
    return response.text
