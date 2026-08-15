# 🛡️ AI Phishing Email Detector

AI-powered phishing email detection system using
Natural Language Processing and Machine Learning.

## Features

- Phishing email detection
- Legitimate email detection
- NLP-based text processing
- TF-IDF vectorization
- Multiple ML algorithms
- Feature engineering
- Flask web application
- Model evaluation
- Confusion matrix
- ROC-AUC analysis

## Machine Learning Models

- Logistic Regression
- Naive Bayes
- Random Forest
- Neural Network

## Technologies

Python
Flask
Scikit-learn
Pandas
NumPy
NLTK
HTML
CSS
JavaScript

## Project Structure

AI-Phishing-Email-Detector/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── dataset/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── best_phishing_model.pkl
│   ├── metadata_features.pkl
│   ├── stop_words.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── phishing_detection.ipynb
│
├── results/
│
├── .gitignore
├── README.md
└── requirements.txt

## Installation

## Clone the repository:

git clone https://github.com/Mandeep141206/AI-Phishing-Email-Detector.git

## Navigate into the project:

cd AI-Phishing-Email-Detector

## Create a virtual environment:

python3 -m venv venv

## Activate it on macOS/Linux:

source venv/bin/activate

## Install dependencies:

pip install -r requirements.txt

▶️ Run the Application

From the project root directory:

python -u app/app.py

The Flask server should start on:

http://127.0.0.1:5000

Open that address in your browser.
...

## Usage
2. Create and Activate Virtual Environment
python -m venv venv

On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Flask Application
python -u app/app.py

The Flask server will start at:

http://127.0.0.1:5000
...

## Results
The project evaluates multiple machine learning approaches for phishing email classification.
| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   99.90% |    99.90% | 99.90% |   99.90% | 99.999% |
| Naive Bayes         |     100% |      100% |   100% |     100% |    100% |
| Random Forest       |     100% |      100% |   100% |     100% |    100% |
| Neural Network      |     100% |      100% |   100% |     100% |    100% |

...

## Screenshots
<img width="1470" height="769" alt="Screenshot 2026-08-15 at 4 09 22 PM" src="https://github.com/user-attachments/assets/1dcca23e-9fb6-44e6-9b41-9af839cd7292" />
<img width="1470" height="774" alt="Screenshot 2026-08-15 at 4 10 16 PM" src="https://github.com/user-attachments/assets/da7d84c0-15ad-47e7-b676-eeeb8a539891" />
<img width="1337" height="671" alt="Screenshot 2026-08-15 at 4 10 32 PM" src="https://github.com/user-attachments/assets/33300194-b7c0-44ca-a68e-50823263acfe" />
<img width="1466" height="777" alt="Screenshot 2026-08-15 at 4 10 49 PM" src="https://github.com/user-attachments/assets/77e68768-c374-4823-b4c9-11fee8472e37" />
<img width="1417" height="705" alt="Screenshot 2026-08-15 at 4 11 06 PM" src="https://github.com/user-attachments/assets/b69f0034-e162-4640-a83f-9a240b3b600a" />
<img width="845" height="569" alt="Screenshot 2026-08-15 at 4 13 54 PM" src="https://github.com/user-attachments/assets/0726bf6c-7a7d-493d-b523-e0b7822b2d9f" />
<img width="803" height="562" alt="Screenshot 2026-08-15 at 4 14 20 PM" src="https://github.com/user-attachments/assets/1be58a26-ba1b-46b8-85f6-038cee3831ab" />
<img width="810" height="557" alt="Screenshot 2026-08-15 at 4 15 15 PM" src="https://github.com/user-attachments/assets/356bee89-a9bb-4fbd-8cf7-1f40b37fb69f" />


## Future Improvements

Possible future enhancements include:

1.Deep-learning-based NLP models
2.Transformer models such as BERT
3.URL reputation analysis
4.Domain-age analysis
5.Email-header analysis
6.Attachment detection
7.Explainable AI
8.Threat-intelligence integration
9.Larger and more diverse datasets
10.Continuous model retraining
11.Docker deployment
12.Cloud deployment
13.Authentication and user management

## Author

Mandeep Singh
AI Phishing Email Detection Project

GitHub:
https://github.com/Mandeep141206
