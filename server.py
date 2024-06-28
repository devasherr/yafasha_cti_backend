from flask import Flask, request, jsonify
from flask_cors import CORS
from markupsafe import escape
from simpletransformers.ner import NERModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertModel
import numpy as np
import pandas as pd
import re
import csv
from io import StringIO
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# !NER
# Path to extracted models
ner_model_directory = "./ner_model/content/outputs/checkpoint-82-epoch-2"

# Initialize the model
ner_model = NERModel('bert', ner_model_directory, use_cuda=False)

@app.post("/ner")
def mainNerPredict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "no text provided"}), 400
    return jsonify(nerPredict(text)), 200
    
def nerPredict(text):
    prediction, model_ouput = ner_model.predict([text])
    return prediction[0]

# !CLASSIFIER
classifier_model_directory = "./classifier/distilbert/content/attack_classifier"

# Load the tokenizer and model
classifier_tokenizer = DistilBertTokenizer.from_pretrained(classifier_model_directory, local_files_only=True)
classifier_model = DistilBertForSequenceClassification.from_pretrained(classifier_model_directory)
@app.post("/classifier")
def main_classifier():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "no text provided"}), 400
    
    return jsonify(classifier(text)), 200
    
def classifier(text):
    # Tokenize the input text
    encoding = classifier_tokenizer(text, return_tensors='pt')

    # Move tensors to the appropriate device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoding = {key: val.to(device) for key, val in encoding.items()}
    classifier_model.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = classifier_model(**encoding)

    # Apply sigmoid to the logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits[0]).cpu().numpy()

    # Define a threshold and get predictions
    threshold = 0.3
    preds = (probs >= threshold).astype(int)

    # Assuming you have the `multilabel` from your previous code to get the class names
    from sklearn.preprocessing import MultiLabelBinarizer

    # Dummy label list to fit the binarizer (replace with actual labels if available)
    label_list = [["Phishing_and_Social_Engineering"], ["APT"], ["Supply_Chain_Attack"], ["Zero-day_Exploit"], ["Ransomware"], ["DDoS"], ["Data_Breach"]]
    multilabel = MultiLabelBinarizer()
    multilabel.fit(label_list)

    # Inverse transform the predictions to get the actual labels
    predicted_labels = multilabel.inverse_transform(preds.reshape(1, -1))
    return predicted_labels[0][0]

# !SENTIMENT
sentiment_model_directory = "./sentiment"

# Load the tokenizer and model
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_directory, local_files_only=True)
sentiment_model = BertTokenizer.from_pretrained(sentiment_model_directory)

@app.post("/sentiment")
def main_sentiment():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "no text provided"}), 400
    return jsonify(sentiment(text))

def sentiment(text):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the class names
    class_names = ['Non-Cyber Related', 'Cyber-Related']

    # Load the model
    class SentimentClassifier(nn.Module):
        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.drop = nn.Dropout(p=0.3)
            self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.pooler_output
            output = self.drop(pooled_output)
            return self.out(output)

    sentiment_model = SentimentClassifier(len(class_names))
    sentiment_model.load_state_dict(torch.load('./sentiment/sentiment_model.pth', map_location=device))
    sentiment_model = sentiment_model.to(device)

    # Load the tokenizer
    sentiment_tokenizer = BertTokenizer.from_pretrained("./sentiment")

    def predict_sentiment(review_text):
        encoded_review = sentiment_tokenizer.encode_plus(
            review_text,
            max_length=160,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        with torch.no_grad():
            output = sentiment_model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
            return class_names[prediction]
        
    return predict_sentiment(text)

#! AUTOMATED
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'the_file' not in request.files:
            return 'No file part'
        
        f = request.files['the_file']
        
        if f.filename == '':
            return 'No selected file'
        
        if not f:
            return jsonify({"error": "no text provided"}), 400
        
        file_content = f.read().decode('utf-8')
        csv_reader = csv.DictReader(StringIO(file_content))
        
        specific_column = 'text'
        extracted_data = []
        
        for row in csv_reader:
            if specific_column in row:
                extracted_data.append(row[specific_column])

        def clean_text(Tweet):
            if isinstance(Tweet, str):
                # Remove hyperlinks
                Tweet = re.sub(r'http\S+', '', Tweet)
                # Remove non-alphanumeric characters
                Tweet = re.sub(r'\W+', ' ', Tweet)
            else:
                Tweet = ""  # If the entry is not a string, return an empty string
            return Tweet

        cleaned_data = [clean_text(data) for data in extracted_data]
        unfiltered_count = len(cleaned_data)

        filtered_data = [data for data in cleaned_data if sentiment(data) == "Cyber-Related"]
        filtered_count = len(filtered_data)

        # get count of classified attacks
        attackTypeCount = defaultdict(int)

        for data in filtered_data:
            attackTypeCount[classifier(data)] += 1

        # get count of ner entities
        nerEntityCount = defaultdict(int)
        for data in filtered_data:
            for ner_result in nerPredict(data):
                for key in ner_result:
                    nerEntityCount[ner_result[key]] += 1    

        return {
                "sentiment": { "related": filtered_count,"unrelated": unfiltered_count - filtered_count},
                "classifier": attackTypeCount,
                "ner": nerEntityCount
        }