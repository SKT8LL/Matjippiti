from chatbot_logic import get_restaurant_recommendation
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("API Key not found!")
else:
    try:
        result = get_restaurant_recommendation(
            api_key=api_key,
            location="강남",
            people=2,
            genre="한식",
            price="3만원",
            notes="조용한 분위기",
            persona_name="백종원"
        )
        print("Success!")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
