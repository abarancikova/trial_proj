from flask import Flask, request, jsonify

#create a Flask object
app = Flask (__name__)


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
    return 0.5

if __name__ == '__main__':
    app.run(debug=True)