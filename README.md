 Initial Plan:
 Design a machine learning-powered resume screening system to improve hiring efficiency while ensuring fairness and transparency.
 Develop models to extract transferable skills, quantify project impact, and detect leadership traits using NLP and pattern recognition.
 Implement Gaussian Mixture Models (GMM), and K-means clustering to classify career types.
 Integrate Google Cloud for real-time deployment and interactive bias audit tools for recruiters.

------------------------------------------------------------------------------------------------------------------------------

🧠 RESUME CLASSIFIER

🚀 Problem Statement
In today's fast-paced recruitment environment, HR professionals and recruiters sift through hundreds of resumes to match candidates to suitable job roles. This manual process is time-consuming, prone to bias, and inefficient, especially when dealing with large volumes of applicants.

⚠️ Challenges Faced:
- Manual resume screening is slow and inconsistent.
- Resumes come in unstructured formats, making automated parsing difficult.
- Assigning the right job role often requires contextual understanding of skills, experience, and keywords.

✅ Solution
Resume Classifier is an intelligent machine learning-based solution that classifies resumes into suitable job roles such as "HR", "Engineer", "Architect", etc. It uses NLP techniques to extract meaningful patterns and classifies resumes based on their content.

🧰 Features
- Automatically reads and parses resumes.
- Classifies into job roles using a trained classification model.
- Uses TF-IDF for feature extraction and various classification models to handle imbalanced datasets.
- Gives higher importance to rare job roles to avoid bias towards majority classes. 
- Supports .pdf format.
