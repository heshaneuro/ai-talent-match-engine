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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)