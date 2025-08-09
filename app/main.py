from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def root():
    return jsonify({
        "message": "AI Talent Match Engine is running!", 
        "status": "active"
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "ai-talent-matcher"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)