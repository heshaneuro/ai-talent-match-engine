from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai


load_dotenv()


def analyze_match_with_ai(job_requirements, candidate_profile):
    """Use OpenAI to analyze how well a candidate matches job requirements"""
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""
        Analyze how well this candidate matches the job requirements:
        
        JOB REQUIREMENTS:
        {job_requirements}
        
        CANDIDATE PROFILE:
        {candidate_profile}
        
        Provide a match score (0-100) and brief explanation.
        Respond in JSON format like: {{"score": 85, "explanation": "Strong match because..."}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


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
def ai_match():
    data = request.get_json()
    
    job_requirements = data.get('job_requirements', 'No requirements provided')
    candidate_profile = data.get('candidate_profile', 'No profile provided')
    
    # Get AI analysis
    ai_result = analyze_match_with_ai(job_requirements, candidate_profile)
    
    return jsonify({
        "job_requirements": job_requirements,
        "candidate_profile": candidate_profile,
        "ai_analysis": ai_result,
        "status": "success"
    })

@app.route('/bulk-match', methods=['POST'])
def bulk_match():
    data = request.get_json()
    
    job_requirements = data.get('job_requirements', '')
    candidates = data.get('candidates', [])
    
    if not candidates:
        return jsonify({"error": "No candidates provided"}), 400
    
    results = []
    for i, candidate in enumerate(candidates):
        candidate_profile = candidate.get('profile', '')
        candidate_name = candidate.get('name', f'Candidate {i+1}')
        
        ai_analysis = analyze_match_with_ai(job_requirements, candidate_profile)
        
        results.append({
            "candidate_name": candidate_name,
            "candidate_profile": candidate_profile,
            "ai_analysis": ai_analysis
        })
    
    return jsonify({
        "job_requirements": job_requirements,
        "total_candidates": len(candidates),
        "matches": results,
        "status": "success"
    })

@app.route('/docs')
def api_docs():
    docs = {
        "API_Documentation": "AI Talent Match Engine",
        "version": "1.0.0",
        "base_url": "https://ai-talent-match-engine.onrender.com",
        "endpoints": {
            "/": "GET - API status check",
            "/health": "GET - Health check",
            "/match": {
                "method": "POST",
                "description": "Match single candidate to job requirements",
                "body": {
                    "job_requirements": "string - Job description and requirements",
                    "candidate_profile": "string - Candidate skills and experience"
                }
            },
            "/bulk-match": {
                "method": "POST", 
                "description": "Match multiple candidates to job requirements",
                "body": {
                    "job_requirements": "string - Job description and requirements",
                    "candidates": [
                        {
                            "name": "string - Candidate name",
                            "profile": "string - Candidate skills and experience"
                        }
                    ]
                }
            }
        }
    }
    return jsonify(docs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)