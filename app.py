import gradio as gr
from api.main import app

def predict_endpoint(age, sexe, temperature, tension_sys, toux, fatigue, maux_tete, region):
    # Appel à votre API
    return f"Diagnostic basé sur les symptômes"

interface = gr.Interface(
    fn=predict_endpoint,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Masculin", "Féminin"], label="Sexe"),
        gr.Number(label="Température"),
        gr.Number(label="Tension Systolique"),
        gr.Checkbox(label="Toux"),
        gr.Checkbox(label="Fatigue"),
        gr.Checkbox(label="Maux de tête"),
        gr.Dropdown(["Dakar", "Ziguinchor", "Saint-Louis"], label="Région")
    ],
    outputs="text"
)

if __name__ == "__main__":
    interface.launch()