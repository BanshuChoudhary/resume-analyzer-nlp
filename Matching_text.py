import streamlit as st
import PyPDF2
import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import wordnet

def upload_pdf():
    upload_file = st.file_uploader("Upload Your PDF file")
    if upload_file is None:
        return ""
    if upload_file is not None:
        pdf = PyPDF2.PdfReader(upload_file)
        text = ""

        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text+=page_text
        return text
resume_text = upload_pdf()
    
# Job desription
if resume_text:
   st.write("length of the text",len(resume_text))
job_description = st.text_area("Paste job description",height=200)

# Analyzing button
if st.button("Analyze"):
    if not resume_text:
        st.warning("please upload your resume first")

    elif not job_description:
        st.warning("please paste your description")

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

        poss = pos_tag(word)
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
            
        lemmatized_word = [lemmatizer.lemmatize(word,word_net_tree(tag)) for word,tag in poss]

        # Stopword
        remove_stop = set(stopwords.words('english'))
        removal_text = [w for w in lemmatized_word if w not in remove_stop]
        resume_final = " ".join(removal_text)


        st.write(resume_final)