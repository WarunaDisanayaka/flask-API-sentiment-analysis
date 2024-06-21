import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Function to load dataset and extract labels
def load_dataset_and_labels(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Extract emotion labels and scores
    emotion_data = df[['review_emotions', 'review_emotion_scores']].iloc[0]
    review_emotions = eval(emotion_data['review_emotions'])
    review_emotions_scores = eval(emotion_data['review_emotion_scores'])
    emotion_labels = {emotion: idx for idx, emotion in enumerate(review_emotions)}

    # Extract theme labels
    theme_labels = {theme: idx for idx, theme in enumerate(df['theme_emotion_X'].unique())}

    # Extract vote labels
    vote_labels = {vote: idx for idx, vote in enumerate(df['vote'].unique())}

    return df, emotion_labels, theme_labels, vote_labels

# Load dataset and labels
file_path = 'new_dataset.csv'
df, emotion_labels, theme_labels, vote_labels = load_dataset_and_labels(file_path)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load trained models
emotion_model = DistilBertForSequenceClassification.from_pretrained('./results-emotion/emotion-checkpoint-best').to(device)
theme_model = DistilBertForSequenceClassification.from_pretrained('./results-theme/theme-checkpoint-best').to(device)
vote_model = DistilBertForSequenceClassification.from_pretrained('./results-vote/vote-checkpoint-best').to(device)

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Function to preprocess text
def preprocess_text(text):
    return str(text).lower().strip() if text else ""

def predict_emotion_and_theme(text):
    clean_text = preprocess_text(text)

    # Emotion Prediction
    emotion_inputs = tokenizer(clean_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    emotion_outputs = emotion_model(**emotion_inputs)
    emotion_logits = emotion_outputs.logits
    emotion_probs = torch.nn.functional.softmax(emotion_logits, dim=1).cpu().detach().numpy()[0]

    # Get top emotions
    review_emotions = list(emotion_labels.keys())
    top_emotions = [{"emotion": emotion, "score": float(emotion_probs[i])} for i, emotion in enumerate(review_emotions)]
    top_emotions_sorted = sorted(top_emotions, key=lambda x: x["score"], reverse=True)

    # Theme Prediction
    theme_inputs = tokenizer(clean_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    theme_outputs = theme_model(**theme_inputs)
    theme_logits = theme_outputs.logits
    predicted_theme_id = np.argmax(theme_logits.cpu().detach().numpy(), axis=1)[0]
    predicted_theme = list(theme_labels.keys())[predicted_theme_id]

    # Vote Prediction
    vote_inputs = tokenizer(clean_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    vote_outputs = vote_model(**vote_inputs)
    vote_logits = vote_outputs.logits
    predicted_vote_id = np.argmax(vote_logits.cpu().detach().numpy(), axis=1)[0]
    predicted_vote = list(vote_labels.keys())[predicted_vote_id]

    # Ensure all values in result are JSON serializable
    result = {
        "theme_emotion_X": str(predicted_theme),  # Convert to string if necessary
        "sentimentScore": round(sum(emotion_probs)),  # Round the sum to the nearest integer
        "vote": int(predicted_vote),  # Convert to int if necessary
        "topEmotions": top_emotions_sorted
    }

    return result



# API Endpoint to predict emotion and theme
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    result = predict_emotion_and_theme(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

