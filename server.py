from flask import Flask, request, jsonify
from markupsafe import escape
from simpletransformers.ner import NERModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

app = Flask(__name__)

# !NER
# Path to extracted models
ner_model_directory = "./ner_model/content/outputs/checkpoint-82-epoch-2"

# Initialize the model
ner_model = NERModel('bert', ner_model_directory, use_cuda=False)

@app.post("/ner")
def nerPredict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "no text provided"}), 400
        
    # Make prediction using the NER model
    prediction, model_ouput = ner_model.predict([text])
    return jsonify(prediction), 200

# !CLASSIFIER
classifier_model_directory = "./classifier/distilbert/content/attack_classifier"

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(classifier_model_directory, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(classifier_model_directory)
@app.post("/classifier")
def classifier():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "no text provided"}), 400
    
    # Tokenize the input text
    encoding = tokenizer(text, return_tensors='pt')

    # Move tensors to the appropriate device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoding = {key: val.to(device) for key, val in encoding.items()}
    model.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**encoding)

    # Apply sigmoid to the logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits[0]).cpu().numpy()

    # Define a threshold and get predictions
    threshold = 0.3
    preds = (probs >= threshold).astype(int)

    # Assuming you have the `multilabel` from your previous code to get the class names
    from sklearn.preprocessing import MultiLabelBinarizer

    # Dummy label list to fit the binarizer (replace with actual labels if available)
    label_list = [["Phishing and Social Engineering"], ["APT"], ["Supply Chain Attack"], ["Zero-day Exploit"], ["Ransomware"], ["DDoS"], ["Data Breach"]]
    multilabel = MultiLabelBinarizer()
    multilabel.fit(label_list)

    # Inverse transform the predictions to get the actual labels
    predicted_labels = multilabel.inverse_transform(preds.reshape(1, -1))
    return jsonify(predicted_labels), 200