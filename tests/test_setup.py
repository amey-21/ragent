# test_setup.py
import chromadb
import sentence_transformers
import langchain
import groq
from dotenv import load_dotenv
import os

load_dotenv()

print("✓ All packages imported")
print(f"✓ Groq key loaded: {'yes' if os.getenv('GROQ_API_KEY') else 'NO - check .env'}")
print(f"✓ ChromaDB version: {chromadb.__version__}")
print(f"✓ LangChain version: {langchain.__version__}")

# Actually test the Groq connection
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in 5 words"}]
)
print(f"✓ Groq connection works: {response.choices[0].message.content}")