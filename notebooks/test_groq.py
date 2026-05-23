import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables depuis .env
load_dotenv()

# Récupérer la clé API
api_key = os.getenv("GROQ_API_KEY")

# Vérifier que la clé existe
if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvée dans .env")
    exit()

print("✅ Clé trouvée!")

# Créer le client Groq
client = Groq(api_key=api_key)

print("Appel à Groq...")

# Premier appel : question simple
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "Tu es un assistant médical sénégalais. Réponds en français simple. Maximum 3 phrases."
        },
        {
            "role": "user",
            "content": "Quels sont les symptômes du paludisme ?"
        }
    ],
    max_tokens=200,
    temperature=0.3
)

# Afficher la réponse
print("\n=== Réponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilisés : {response.usage.total_tokens}")