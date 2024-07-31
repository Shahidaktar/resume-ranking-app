from flask import Flask, request
import requests
import fitz  
from io import BytesIO
import re
import pickle
from flask_cors import CORS 

app = Flask(__name__)
CORS(app,origins="*")
rfc_job_recommendation = pickle.load(open('rfc_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('tfidf_vectorizer_job_recommendation.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rfc_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

def extract_text_from_pdf_url(url):
    try:
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with 'http://' or 'https://'")

        
        response = requests.get(url)
        response.raise_for_status()  
        pdf_document = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        pdf_document.close()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_skills_from_resume(text, skills_list):
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

@app.route('/')
def home():
    return "it works"

@app.route('/recommend-job', methods=['POST'])
def pred():
    data = request.get_json()
    if( data):    
        extracted_text = extract_text_from_pdf_url(data['url']) 
        recommended_job = job_recommendation(extracted_text)
        return {"job":recommended_job}
    else:
          return "Error"
    
@app.route('/score', methods=['POST'])
def score():
    data = request.get_json()
    if( data):    
        extracted_text = extract_text_from_pdf_url(data['url']) 
        cleaned_resume=cleanResume(extracted_text)
        extracted_skills = extract_skills_from_resume(cleaned_resume, data['skills'])
        return {"score":int(float((len(extracted_skills)/len(data['skills']))*100))}
    else:
          return "Error"

if __name__ == '__main__':
    app.run(debug=True)