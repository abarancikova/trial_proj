import torch
from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer

# Create a Flask object
app = Flask(__name__)

# Load the fine-tuned BERT model
model = BertForSequenceClassification.from_pretrained('fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Manually add CORS headers
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow the Content-Type header
    return response

app.after_request(add_cors_headers)

@app.route('/process-argument', methods=['POST'])

# Process the argument and call the evaluate function
# Returns the quality score to the js file, throws an exeption if uncessful
def process_argument():
    try:
        argument_text = request.json.get('argument')
        quality_score = evaluate_argument(argument_text)
        return jsonify({'quality_score': quality_score}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Evaluates argument and fits the score to the range 0 to 1
# Returns the value of the quality score
def evaluate_argument(argument_text):
    # Tokenize the argument
    inputs = tokenizer(argument_text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract predicted quality score and apply sigmoid
    logits = outputs.logits
    quality_score = torch.sigmoid(logits).item()
    
    return (quality_score)

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
