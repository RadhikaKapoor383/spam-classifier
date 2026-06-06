from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Same preprocessing as your notebook
def clean_text(text):
    text = text.lower()                            # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)    # remove URLs
    text = re.sub(r'\d+', '', text)                # remove numbers
    text = re.sub(r'[^\w\s]', '', text)            # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    result = '🚨 Spam!' if prediction == 1 else '✅ Not Spam!'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)