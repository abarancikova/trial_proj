from flask import Flask, request, jsonify


#create a Flask object
app = Flask (__name__)

#add cors headers
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow the Content-Type header
    return response

app.after_request(add_cors_headers)


@app.route('/process-argument', methods=['POST'])
def process_argument():
    print("here")
    try:
        argument_text = request.json.get('argument')
        quality_score = evaluate_argument(argument_text)
        

        return jsonify({'quality_score': quality_score}), 200
    
    except Exception as e:
        
        return jsonify({'error':str(e)}), 500

def evaluate_argument(argument_text):
    #TODO evaluate arguement

    #return score
    return 0.4

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)