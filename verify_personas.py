
import os
from dotenv import load_dotenv
from chatbot_logic import get_restaurant_recommendation

# Load environment variables
load_dotenv()

# API Key check (optional, as rest_rec files have hardcoded failover, but good to have)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not found in env, relying on rest_rec module's hardcoded key.")
    api_key = "dummy_key"

print("--- Testing Ahn Sung-jae Persona ---")
try:
    response_ahn = get_restaurant_recommendation(
        api_key=api_key,
        location="SEOUL",
        people=2,
        genre="KOREAN",
        price="30000",
        notes="Testing Ahn Persona",
        persona_name="안성재"
    )
    print("Response (Ahn):")
    print(response_ahn[:500] + "...") # Print first 500 chars
except Exception as e:
    print(f"Error testing Ahn: {e}")

print("\n--- Testing Baek Jong-won Persona ---")
try:
    response_baik = get_restaurant_recommendation(
        api_key=api_key,
        location="SEOUL",
        people=4,
        genre="CHINESE",
        price="10000",
        notes="Testing Baek Persona",
        persona_name="백종원"
    )
    print("Response (Baik):")
    print(response_baik[:500] + "...") # Print first 500 chars
except Exception as e:
    print(f"Error testing Baek: {e}")
