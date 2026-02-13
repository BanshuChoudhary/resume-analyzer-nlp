# Resume Analyzer (NLP Based)

A Python-based Resume Analyzer that evaluates a resume against a Job Description using Natural Language Processing (NLP) techniques.

## Features

- PDF Resume Text Extraction (pdfplumber)
- Text Cleaning & Preprocessing
- Tokenization
- POS Tagging
- Lemmatization
- Stopword Removal
- TF-IDF Vectorization (Unigram + Bigram)
- Cosine Similarity Scoring
- ATS Compatibility Percentage

## Tech Stack

- Python
- Streamlit
- NLTK
- Scikit-learn
- pdfplumber

## How It Works

1. Extracts text from uploaded resume PDF  
2. Cleans and preprocesses text  
3. Applies NLP techniques (POS tagging + Lemmatization)  
4. Converts resume & job description into TF-IDF vectors  
5. Calculates similarity using Cosine Similarity  
6. Displays ATS compatibility score  

##  How to Run Locally

Clone the repository:

