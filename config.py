import os
import json

def load_api_keys(config_file="config.json"):
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('OPENAI_API_KEY'), config.get('GEMINI_API_KEY')
        except Exception as e:
            print(f"Error leyendo config.json: {e}")
    OPENAI_API_KEY = "REMOVIDOproj-2wlRQZ-r6tys-B3zsTU_e04PGxdq9SwTP0oVnzWuMsHF2XFnBTV06-xtusGSvpp-bgtSg93RItT3BlbkFJO5OzCdGQWz05C7OetoB6Ld86OhJLp9ZfC5qVM47I_oZj58l3YujZCqFa86_Rg-ugDg5IOCjdsA"
    GEMINI_API_KEY = "AIzaSyDM3BcG2YZtZJMgi6IAlAb19r7t0ouxuDQ"
    return OPENAI_API_KEY, GEMINI_API_KEY

def get_openai_client(api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    return client

def get_gemini_model(api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    return model