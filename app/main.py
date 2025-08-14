from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai

import json as json_module
import re

import tempfile
import os
from pathlib import Path


load_dotenv()


app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", methods="*", supports_credentials=True)



def convert_structured_data_to_html(structured_data):
    """Convert structured PDF data to HTML format like your visualization script"""
    html_content = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 2rem; }",
        ".section { margin: 0.5rem 0; }",
        ".large-text { font-size: 1.1em; font-weight: bold; }",
        ".medium-text { font-size: 1em; }",
        ".small-text { font-size: 0.9em; }",
        "</style>",
        "</head><body>"
    ]

    for section in structured_data:
        if section['text'].strip():  # Only add non-empty sections
            # Determine text size class based on font size
            size_class = 'medium-text'
            if section['font_size'] > 15:
                size_class = 'large-text'
            elif section['font_size'] < 10:
                size_class = 'small-text'

            html_content.append(
                f'<div class="section {size_class}">{section["text"]}</div>'
            )

    html_content.append("</body></html>")
    return "\n".join(html_content)



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
        "API_Documentation": "AI Talent Match Engine - Cybersecurity Edition",
        "version": "1.1.0",
        "description": "AI-powered cybersecurity talent matching with CSV upload support",
        "base_url": "https://ai-talent-match-engine.onrender.com",
        
        "endpoints": {
            "GET /": {
                "description": "API status check",
                "response": "Service status message"
            },
            
            "GET /health": {
                "description": "System health check",
                "response": "Health status confirmation"
            },
            
            "POST /match": {
                "description": "Match single candidate to security job requirements",
                "content_type": "application/json",
                "example_request": {
                    "job_requirements": "GRC expert with audit experience",
                    "candidate_profile": "Senior Security Analyst with 7 years GRC experience"
                },
                "example_response": {
                    "ai_analysis": "Score and explanation",
                    "status": "success"
                }
            },
            
            "POST /bulk-match": {
                "description": "Match multiple security experts to job requirements",
                "content_type": "application/json",
                "example_request": {
                    "job_requirements": "Red Team specialist for penetration testing",
                    "candidates": [
                        {"name": "John Doe", "profile": "Penetration tester with 5 years experience"}
                    ]
                }
            },
            
            "POST /process-candidate-data": {
                "description": "Process cybersecurity experts from structured JSON data",
                "content_type": "application/json",
                "use_case": "API integration with existing systems",
                "note": "Renamed from /upload-candidates for clarity",
                "csv_fields_supported": [
                    "Full Name",
                    "Job Title", 
                    "Country",
                    "Years of Experience",
                    "Main Category (GRC/Red Team/Blue Team/System Security)",
                    "Specialization details"
                ]
            },
            
            "POST /upload-csv": {
                "description": "Upload actual CSV file and process cybersecurity experts",
                "content_type": "multipart/form-data",
                "use_case": "Direct CSV file upload from CyberTalents exports",
                "form_fields": {
                    "file": "CSV file (required)",
                    "job_requirements": "Job description text (optional)"
                },
                "supported_format": "CyberTalents expert registration format"
            },
            
            "POST /match-analytics": {
                "description": "Cybersecurity talent pool analytics and insights",
                "content_type": "application/json",
                "provides": [
                    "Match score distribution",
                    "Cybersecurity specialty breakdown",
                    "Experience level analysis", 
                    "Geographic distribution",
                    "Top candidates ranking"
                ]
            }
        },
        
        "how_to_upload_csv": {
            "step_1": "Prepare CSV file with cybersecurity expert data",
            "step_2": "Use POST to /upload-csv with multipart/form-data",
            "step_3": "Include 'file' field with your CSV",
            "step_4": "Optionally add 'job_requirements' text field",
            "step_5": "Receive AI-powered matching results instantly"
        },
        
        "supported_specialties": [
            "GRC (Governance, Risk, Compliance)",
            "Red Team (Penetration Testing, Ethical Hacking)",
            "Blue Team (SOC, DFIR, Threat Hunting)",
            "System Security (Network, Cloud, Infrastructure)"
        ],
        
        "contact": "API ready for integration - built for cybersecurity recruitment"
    }
    
    # Use Flask's jsonify with proper formatting
    from flask import Response
    import json
    
    # Return properly formatted JSON
    return Response(
        json.dumps(docs, indent=2, ensure_ascii=False),
        mimetype='application/json',
        headers={'Content-Type': 'application/json; charset=utf-8'}
    )

@app.route('/process-candidate-data', methods=['POST'])
def process_candidate_data():
    """Process cybersecurity expert data from JSON format (formerly upload-candidates)"""
    # Keep all the existing code exactly the same, just rename the function
    try:
        data = request.get_json()
        
        job_requirements = data.get('job_requirements', '')
        csv_data = data.get('csv_data', [])
        
        if not csv_data:
            return jsonify({"error": "No candidate data provided"}), 400
        
        results = []
        for candidate_row in csv_data:
            # Build comprehensive cybersecurity profile
            profile_parts = []
            
            # Basic info
            name = candidate_row.get('Full Name', '').strip()
            job_title = candidate_row.get('Job Title', '').strip()
            country = candidate_row.get('Country', '').strip()
            experience = candidate_row.get('Years of Experience', '').strip()
            
            if job_title:
                profile_parts.append(f"{job_title}")
            if country:
                profile_parts.append(f"from {country}")
            if experience:
                profile_parts.append(f"with {experience} experience")
            
            # Main cybersecurity category
            main_category = candidate_row.get('Kindly select One Main Category you would like to apply for', '').strip()
            if main_category:
                profile_parts.append(f"specializes in {main_category}")
            
            # Specific specializations
            grc_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in GRC category', '').strip()
            redteam_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in Red Team category', '').strip()
            blueteam_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in Blue Team category', '').strip()
            syssec_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in (System Security) category', '').strip()
            
            if grc_spec:
                profile_parts.append(f"GRC expertise: {grc_spec}")
            if redteam_spec:
                profile_parts.append(f"Red Team skills: {redteam_spec}")
            if blueteam_spec:
                profile_parts.append(f"Blue Team skills: {blueteam_spec}")
            if syssec_spec:
                profile_parts.append(f"System Security: {syssec_spec}")
            
            # Training experience
            training_exp = candidate_row.get('How many years of experience do you have in delivering cybersecurity training', '').strip()
            if training_exp and training_exp != '':
                profile_parts.append(f"Training experience: {training_exp}")
            
            # Combine into full profile
            candidate_profile = ". ".join(profile_parts) if profile_parts else "Limited profile information available"
            
            # Get AI analysis
            ai_analysis = analyze_match_with_ai(job_requirements, candidate_profile)
            
            results.append({
                "candidate_name": name or "Unknown",
                "candidate_profile": candidate_profile,
                "raw_data": candidate_row,
                "ai_analysis": ai_analysis
            })
        
        return jsonify({
            "job_requirements": job_requirements,
            "total_candidates": len(csv_data),
            "matches": results,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/match-analytics', methods=['POST'])
def cybersecurity_analytics():
    """Analyze cybersecurity talent pool with industry-specific insights"""
    try:
        data = request.get_json()
        
        job_requirements = data.get('job_requirements', '')
        csv_data = data.get('csv_data', [])
        
        if not csv_data:
            return jsonify({"error": "No candidate data provided"}), 400
        
        # Process all candidates and extract scores
        scores = []
        matches = []
        categories = {'GRC': 0, 'Red Team': 0, 'Blue Team': 0, 'System Security': 0, 'Other': 0}
        experience_levels = {'Junior (0-2 years)': 0, 'Mid (3-5 years)': 0, 'Senior (5+ years)': 0}
        countries = {}
        
        for candidate_row in csv_data:
            # Build profile (same logic as upload endpoint)
            profile_parts = []
            name = candidate_row.get('Full Name', '').strip()
            job_title = candidate_row.get('Job Title', '').strip()
            country = candidate_row.get('Country', '').strip()
            experience = candidate_row.get('Years of Experience', '').strip()
            main_category = candidate_row.get('Kindly select One Main Category you would like to apply for', '').strip()
            
            # Build profile
            if job_title:
                profile_parts.append(f"{job_title}")
            if country:
                profile_parts.append(f"from {country}")
            if experience:
                profile_parts.append(f"with {experience} experience")
            if main_category:
                profile_parts.append(f"specializes in {main_category}")
            
            candidate_profile = ". ".join(profile_parts) if profile_parts else "Limited information"
            
            # Get AI analysis and extract score
            ai_analysis = analyze_match_with_ai(job_requirements, candidate_profile)
            try:
                clean_response = ai_analysis.replace('```json\n', '').replace('\n```', '').strip()
                analysis_data = json_module.loads(clean_response)
                score = analysis_data.get('score', 0)
                scores.append(score)
            except:
                scores.append(0)
            
            matches.append({
                "candidate_name": name or "Unknown",
                "score": scores[-1] if scores else 0,
                "ai_analysis": ai_analysis
            })
            
            # Analytics data collection
            # Category distribution
            if 'GRC' in main_category:
                categories['GRC'] += 1
            elif 'Red Team' in main_category:
                categories['Red Team'] += 1
            elif 'Blue Team' in main_category:
                categories['Blue Team'] += 1
            elif 'System Security' in main_category:
                categories['System Security'] += 1
            else:
                categories['Other'] += 1
            
            # Experience level
            if any(x in experience.lower() for x in ['0-2', '1-2', '0-3']):
                experience_levels['Junior (0-2 years)'] += 1
            elif any(x in experience.lower() for x in ['3-5', '3-7', '4-6']):
                experience_levels['Mid (3-5 years)'] += 1
            else:
                experience_levels['Senior (5+ years)'] += 1
            
            # Country distribution
            if country:
                countries[country] = countries.get(country, 0) + 1
        
        # Calculate score analytics
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            excellent = len([s for s in scores if s >= 80])
            good = len([s for s in scores if 60 <= s < 80])
            fair = len([s for s in scores if 40 <= s < 60])
            poor = len([s for s in scores if s < 40])
        else:
            avg_score = max_score = min_score = 0
            excellent = good = fair = poor = 0
        
        # Sort matches by score
        top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
        
        return jsonify({
            "job_requirements": job_requirements,
            "analytics": {
                "total_candidates": len(csv_data),
                "average_match_score": round(avg_score, 1),
                "highest_score": max_score,
                "lowest_score": min_score,
                "score_distribution": {
                    "excellent_80_plus": excellent,
                    "good_60_79": good,
                    "fair_40_59": fair,
                    "poor_below_40": poor
                },
                "cybersecurity_specialties": categories,
                "experience_distribution": experience_levels,
                "geographic_distribution": countries
            },
            "top_5_matches": top_matches,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import io
import csv

@app.route('/upload-csv', methods=['POST'])
def upload_csv_file():
    """Upload actual CSV file and process cybersecurity expert data"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get job requirements from form data
        job_requirements = request.form.get('job_requirements', 'General cybersecurity position')
        
        # Read and parse CSV file
        if file and file.filename.endswith('.csv'):
            # Read file content
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_reader = csv.DictReader(stream, delimiter=',')
            
            # Convert CSV rows to list of dictionaries
            csv_data = []
            debug_printed = False
            for row in csv_reader:
                # DEBUG: Print column names and first row (only once)
                if not debug_printed:
                    print("=" * 50)
                    print("DEBUG - CSV PARSING INFO")
                    print("=" * 50)
                    print("Column names found:", list(row.keys()))
                    print("Number of columns:", len(row.keys()))
                    print("First row raw data:")
                    for key, value in row.items():
                        print(f"  '{key}' = '{value}'")
                    print("=" * 50)
                    debug_printed = True
                # Clean up the row data
                clean_row = {}
                for key, value in row.items():
                    if key and value:  # Skip empty keys/values
                        clean_row[key.strip()] = str(value).strip()
                if clean_row:  # Only add non-empty rows
                    csv_data.append(clean_row)
            
            if not csv_data:
                return jsonify({"error": "No valid data found in CSV file"}), 400
            
            # Process using the same logic as process-candidate-data
            results = []
            for candidate_row in csv_data:
                # Build comprehensive cybersecurity profile (same logic as before)
                profile_parts = []
                
                # Basic info
                name = candidate_row.get('Full Name', '').strip()
                job_title = candidate_row.get('Job Title', '').strip()
                country = candidate_row.get('Country', '').strip()
                experience = candidate_row.get('Years of Experience', '').strip()
                
                if job_title:
                    profile_parts.append(f"{job_title}")
                if country:
                    profile_parts.append(f"from {country}")
                if experience:
                    profile_parts.append(f"with {experience} experience")
                
                # Main cybersecurity category
                main_category = candidate_row.get('Kindly select One Main Category you would like to apply for', '').strip()
                if main_category:
                    profile_parts.append(f"specializes in {main_category}")
                
                # Specific specializations
                grc_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in GRC category', '').strip()
                redteam_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in Red Team category', '').strip()
                blueteam_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in Blue Team category', '').strip()
                syssec_spec = candidate_row.get('Kindly select the Specialization you would like to apply for in (System Security) category', '').strip()
                
                if grc_spec:
                    profile_parts.append(f"GRC expertise: {grc_spec}")
                if redteam_spec:
                    profile_parts.append(f"Red Team skills: {redteam_spec}")
                if blueteam_spec:
                    profile_parts.append(f"Blue Team skills: {blueteam_spec}")
                if syssec_spec:
                    profile_parts.append(f"System Security: {syssec_spec}")
                
                # Training experience
                training_exp = candidate_row.get('How many years of experience do you have in delivering cybersecurity training', '').strip()
                if training_exp and training_exp != '':
                    profile_parts.append(f"Training experience: {training_exp}")
                
                # Combine into full profile
                candidate_profile = ". ".join(profile_parts) if profile_parts else "Limited profile information available"
                
                # Get AI analysis
                ai_analysis = analyze_match_with_ai(job_requirements, candidate_profile)
                
                results.append({
                    "candidate_name": name or "Unknown",
                    "candidate_profile": candidate_profile,
                    "ai_analysis": ai_analysis
                })
            
            return jsonify({
                "job_requirements": job_requirements,
                "filename": file.filename,
                "total_candidates": len(csv_data),
                "matches": results,
                "status": "success"
            })
        
        else:
            return jsonify({"error": "Please upload a CSV file"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/extract-cv-experience', methods=['POST'])
def extract_cv_experience():
    """Extract cybersecurity experience from uploaded CV PDF"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No CV file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Please upload a PDF file"}), 400
        
        # Check file size (5MB limit)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if file_size > 5 * 1024 * 1024:  # 5MB in bytes
            return jsonify({"error": "File size too large. Maximum 5MB allowed."}), 400
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_pdf_path = temp_file.name
        
        try:
            # Extract structured data from PDF
            structured_data = extract_pdf_to_structured_data(temp_pdf_path)
            
            if not structured_data:
                return jsonify({"error": "Failed to extract content from PDF. Please ensure it's a valid PDF with text content."}), 400
            
            # Convert to HTML format (matching your original workflow)
            html_content = convert_structured_data_to_html(structured_data)
            
            if not html_content or len(html_content.strip()) < 200:
                return jsonify({"error": "PDF appears to contain insufficient content for analysis."}), 400
            
            # TODO: Add OpenAI experience extraction from HTML (Phase 3)
            return jsonify({
                "filename": file.filename,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "html_preview": html_content[:500] + "..." if len(html_content) > 500 else html_content,
                "html_length": len(html_content),
                "status": "html_generated",
                "message": "PDF converted to HTML successfully. OpenAI analysis coming in Phase 3."
            })
            
        finally:
            # Clean up temporary file after short delay (keep for debugging as requested)
            import threading
            def cleanup_file():
                import time
                time.sleep(300)  # Keep file for 5 minutes
                try:
                    os.unlink(temp_pdf_path)
                except:
                    pass
            
            threading.Thread(target=cleanup_file, daemon=True).start()
            
    except Exception as e:
        return jsonify({"error": f"Failed to process CV: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)