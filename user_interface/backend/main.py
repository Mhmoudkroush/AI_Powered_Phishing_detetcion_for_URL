import io
from concurrent.futures.thread import ThreadPoolExecutor
from flask import Flask , request, jsonify, Response
import re # check if domain is IP address
from urllib.parse import urlparse, urlunparse
import torch
import torch.nn as nn
from flask_cors import CORS
import socket
import pandas as pd
from werkzeug.utils import secure_filename
import os
from phishing_url_generator import generate_urls
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, bidirectional=True, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional, 
                            dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        hidden = self.dropout(hidden)
        return self.fc(hidden)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
char_to_idx = torch.load('char_to_idx.pth') 
# parameters for LSTM model
vocab_size = len(char_to_idx) + 1
embedding_dim = 128
hidden_dim = 128
output_dim = 1
num_layers = 2
dropout = 0.5 
max_length = 100

baseModel = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=num_layers, bidirectional=True, dropout=0.5)
try:
    state_dict = torch.load("phishing_detector_lstm_fold1.pth", map_location=device)
    baseModel.load_state_dict(state_dict)
    baseModel.to(device)
    baseModel.eval()
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")


@app.route('/')
def home():
    return "Welcome to the Phishing Detection System!"

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


def resolve_domain(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname and not re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname):
            ip = socket.gethostbyname(hostname)
            new_netloc = ip
            if parsed.port: new_netloc += f':{parsed.port}'
            parsed = parsed._replace(netloc=new_netloc)
            return urlunparse(parsed)
        return url
    except:
        return url


def predict_url_logic(url, preprocess=True):
    try:


        def normalize_url(url):
           url = url.lower().strip()

           parsed = urlparse(url)

           netloc = parsed.netloc

           if netloc == "":
               netloc = parsed.path
               path = ""
           else:
               path = parsed.path
    
           if netloc.startswith("www."):
                netloc = netloc[4:]

           return netloc + path
        
        processed_url = normalize_url(url) if preprocess else url

        sequence = [char_to_idx.get(char, 0) for char in processed_url]

        if len(sequence) < max_length:
            padded = sequence + [0] * (max_length - len(sequence))
        else:
            padded = sequence[:max_length]
        
        input_tensor = torch.LongTensor([padded]).to(device)
        
        with torch.no_grad():
            output = baseModel(input_tensor)
            probability = torch.sigmoid(output).item()
        
        print(f"Checking: {url[:30]}... | Prob: {probability:.4f}")
        
        return {
            'original_url': url,
            'phishing_probability': probability,
            'prediction': 'phishing' if probability > 0.5 else 'legitimate'
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {'original_url': url, 'prediction': 'error', 'phishing_probability': 0}
@app.route('/detect', methods=['POST'])
def detect_phishing():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    result = predict_url_logic(url, preprocess=True)
    return jsonify(result)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        df = pd.read_csv(file)
        if 'url' not in df.columns:
            return jsonify({'error': 'CSV must have "url" column'}), 400
        
        urls = [url for url in df['url'] if pd.notna(url)]

    
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda u: predict_url_logic(u, preprocess=True), urls))
        return jsonify({
            'message': 'Success',
            'results': results,
            'total': len(results)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/generate_phishing', methods=['GET'])
def generate_phishing():
    try:
        
        urls = generate_urls(1000) 
        results = [predict_url_logic(url, preprocess=False) for url in urls]
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)