import streamlit as st
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import pdfplumber

import nltk

# ---- NLTK Downloads for Streamlit Cloud ----
resources = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for resource in resources:
    try:
        nltk.download(resource)
    except:
        pass


st.set_page_config(page_title="Resume-Analyzer",
                   page_icon="📄",
                   layout="centered"
                   )

st.title("Resume Analyzer")
st.write("Evaluate your resume against a job description using NLP-based similarity scoring.")

with st.sidebar:
    st.title("About This Application")
    st.subheader("Resume - Job Description Matching System")
    st.write("""
This tool analyzes your resume against a given job description using:

-> Text preprocessing (Tokenization, POS tagging, Lemmatization)  
-> TF-IDF Vectorization with bigram support  
-> Cosine Similarity scoring  

Upload your resume and paste a job description to receive an ATS compatibility score.
            """)

# Uploading PDF
def upload_pdf():
    upload_file = st.file_uploader("Upload Your Resume (PDF Format)")
    if upload_file is None:
        return ""
    
    text = ""
    
    with pdfplumber.open(upload_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

resume_text = upload_pdf()

if resume_text:
   st.write("Extracted Resume Text Length:", len(resume_text))

job_description = st.text_area("Paste the Job Description Here", height=200)

# Analyze button
if st.button("Analyze Resume"):

    if not resume_text:
        st.warning("Please upload your resume before running the analysis.")

    elif not job_description:
        st.warning("Please paste the job description to proceed with analysis.")

    else:
        # cleaning the text
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9+#.\s]","",text)
            text = re.sub(r"\s+"," ",text).strip()
            return text

        clean = clean_text(resume_text)

        # Tokenize the cleaned text
        word = word_tokenize(clean)

        # Pos_tag + Lemmatization
        Pos_tag = pos_tag(word)
        lemmatizer = WordNetLemmatizer()

        def word_net_tree(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
            
        lemmatized_word = [lemmatizer.lemmatize(word,word_net_tree(tag)) for word,tag in Pos_tag]

        # Stopword
        remove_stop = set(stopwords.words('english'))
        removal_text = [w for w in lemmatized_word if w not in remove_stop]
        resume_final = " ".join(removal_text)

        vectorize = TfidfVectorizer(ngram_range=(1,2),lowercase=True,max_features=5000)

        # JD processing
        jd_clean = clean_text(job_description)
        jd_token = word_tokenize(jd_clean)
        jd_pos = pos_tag(jd_token)

        jd_lemmatizer = [
           lemmatizer.lemmatize(w,word_net_tree(tag))
           for w, tag in jd_pos
        ]

        jd_token = [
            w for w in jd_lemmatizer if w not in remove_stop
        ]

        jd_final = " ".join(jd_token)

        # Connection of resume and JD
        vector = vectorize.fit_transform([resume_final,jd_final])

        # finding similarity via cosine similarity
        similarity = cosine_similarity(vector[0], vector[1])[0][0]
        ats_score = round(similarity * 100, 2)

        st.subheader("ATS Compatibility Score")
        st.write(ats_score,"%")
