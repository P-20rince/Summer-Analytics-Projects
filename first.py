from flask import Flask, request, jsonify
import openai

openai.api_key = "sk-proj-dTWCE8aprB1H7FGh784eT3BlbkFJ0qRpYNl4guf2VwoqUWUZ"

app = Flask(__name__)

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    prompt = data['prompt']
    response = chat_with_gpt(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
