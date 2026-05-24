from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
import os
from dotenv import load_dotenv
import joblib
app = FastAPI()

# ⭐ CORS DOIT ÊTRE ICI, AVANT TOUTE ROUTE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger .env
load_dotenv()

# Créer le client Groq
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Client Groq initialisé")
else:
    print("⚠️ GROQ_API_KEY non trouvée")

# Charger .env
load_dotenv()

# Créer le client Groq au démarrage
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Client Groq initialisé")
else:
    print("⚠️ GROQ_API_KEY non trouvée - /explain sera désactivé")

# Schéma Pydantic pour /explain
class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic prédit par le modèle")
    probabilite: float = Field(..., description="Probabilité (0.0 à 1.0)")
    age: int
    sexe: str
    temperature: float
    region: str

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en français")
    modele_llm: str = Field(default="llama-3.1-8b-instant")

# System prompt médical
SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple, comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même – tu expliques uniquement le diagnostic fourni."""

# Endpoint POST /explain
@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""
    
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    
    return ExplainOutput(explication=explication)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas Pydantic ---

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# --- Chargement du modèle ---

print("Chargement du modèle...")
model = joblib.load("/app/models/model.pkl")
le_sexe = joblib.load("/app/models/encoder_sexe.pkl")
le_region = joblib.load("/app/models/encoder_region.pkl")
feature_cols = joblib.load("/app/models/feature_cols.pkl")-

print(f"Modèle chargé : {list(model.classes_)}")

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}")
    
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}")
    
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        region_enc
    ]])
    
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())
    confiance = ("haute" if proba_max >= 0.7
                 else "moyenne" if proba_max >= 0.4
                 else "faible")
    
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoïde. Consultation nécessaire.",
        "sain": "Pas de pathologie détectée."
    }
    
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )
class ModelInfo(BaseModel):
    type: str
    nombre_arbres: int
    classes: list
    nombre_features: int

@app.get("/model-info", response_model=ModelInfo)
def get_model_info():
    return ModelInfo(
        type=type(model).__name__,
        nombre_arbres=model.n_estimators,
        classes=list(model.classes_),
        nombre_features=model.n_features_in_
    )
@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""
    
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    
    return ExplainOutput(explication=explication)
