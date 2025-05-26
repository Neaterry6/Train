
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests

app = Flask(__name__)

Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

Dictionary-based response system
responses = {
    0: "I'm not sure what you mean.",
    1: "I think I understand what you're saying.",
}

def predict(input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    prediction = outputs.logits.argmax(-1).item()
    return prediction

def search(query):
    # Using a simple search API, replace with your preferred search engine
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    return response.text

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    input_text = request.json["input_text"]
    prediction = predict(input_text)
    return jsonify({"prediction": prediction})

@app.route("/chat", methods=["POST"])
def chat():
    input_text = request.json["input_text"]
    if input_text.startswith("research"):
        query = input_text.replace("research", "").strip()
        results = search(query)
        return jsonify({"response": f"Research results for {query}: {results}"})
    else:
        prediction = predict(input_text)
        response = responses.get(prediction, "I'm not sure how to respond.")
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
