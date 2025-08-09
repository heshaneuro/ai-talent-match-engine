from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

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

@app.route('/match', methods=['POST'])
def simple_match():
    data = request.get_json()
    job_title = data.get('job_title', 'No title provided')
    
    # For now, return a simple response - we'll add OpenAI logic next
    return jsonify({
        "job_title": job_title,
        "message": "Matching endpoint is working!",
        "status": "ready for AI integration"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)