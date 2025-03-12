import joblib
import gradio as gr
import numpy as np
import os
import random

MODEL_PATH = "employability_model_selected.joblib"
ENCODER_PATH = "label_encoder_fixed.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

success_message = "😊 Awesome, {name}! You're employable and ready to shine! 🚀"
failure_message = "😞 Keep pushing, {name}! Improve your skills and try again! 💪"

def predict_employability(name, degree, manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness):
    input_data = np.array([[manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness]])
    prediction = model.predict(input_data)[0]
    result = label_encoder.inverse_transform([prediction])[0]
    
    return success_message.format(name=name) if result == "Employable" else failure_message.format(name=name)

iface = gr.Interface(
    fn=predict_employability,
    inputs=[
        gr.Textbox(label="👤 Enter Your Name"),
        gr.Slider(1, 5, step=1, label="🗣️ Manner of Speaking"),
        gr.Slider(1, 5, step=1, label="💬 Self-Confidence"),
        gr.Slider(1, 5, step=1, label="🎤 Ability to Present Ideas"),
        gr.Slider(1, 5, step=1, label="🧠 Communication Skills"),
        gr.Slider(1, 5, step=1, label="⚡ Mental Alertness")
    ],
    outputs=gr.Textbox(label="🎉 Your Career Verdict!"),
    title="Employability Challenge: Are You Job-Ready?",
    description="Welcome to the Employability Quest! Rate your skills from 1-5 and see if you conquer the employability challenge! 🚀",
    submit_btn="🔮Reveal Your Potential!🔥"
)

if __name__ == "__main__":
    iface.launch(share=True)
