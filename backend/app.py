from flask import Flask, request, jsonify
from transformers import BertModel
from transformers import BertModel, BertTokenizer
import torch
from load_data import load_dataset

#create a Flask object
app = Flask (__name__)

#use Bert base uncased as the pre-trained model
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset=load_dataset('backend/dataset/arg_quality_rank_30k.csv')



#add cors headers
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow the Content-Type header
    return response

app.after_request(add_cors_headers)


@app.route('/process-argument', methods=['POST'])
def process_argument():
    try:
        argument_text = request.json.get('argument')
        quality_score = evaluate_argument(argument_text)
        

        return jsonify({'quality_score': quality_score}), 200
    
    except Exception as e:
        
        return jsonify({'error':str(e)}), 500

def evaluate_argument(argument_text):
     # convert argument text to tokens
    tokens = tokenizer.encode(argument_text, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(tokens)
    
    # extract pooled output from BERT
    pooled_output = outputs.pooler_output
    

    linear_layer = torch.nn.Linear(pooled_output.size(-1), 1)
    sigmoid = torch.nn.Sigmoid()
    quality_score = sigmoid(linear_layer(pooled_output))

    # convert tensor to float
    quality_score = quality_score.item()

    print("Quality score:", quality_score)
    return quality_score



if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)