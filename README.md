# Fake_Job_Prediction
Build two models with Naive Bayes, SGD to identify fake job post on social media. <br />
You can use this model to predict any potential fake job post on social media. <br />
Tool used: Python, Scikit-learn, Pandas. <br />
Dataset: job_train.csv  <br />

In the job_train dataset, there are 7 features and 1 label.  
Features are:  
title (the job title of this post - ex: Web Designer), location(the working place of this post - ex: US, TX, USA Southwest), description(the detail of this job description)  
requirement(the criteria to fulfill the basic skills in this job)  
telecomuting(does the employer leave the phone number, ex: 1 or 0)  
has_company_logo(does this company have a logo? ex: 1 or 0)  
has_questions(do they leave some questions on this post, ex: 1 or 0).  
The label is fraudulent(Is this post an actual job post, or is it fake? ex: 1 or 0).  
In this model, TfidVectorizer was implemented to turn the effective features from words into vectors, and SGDClassifier was implemented to train this model with the Grid Search to find the best parameters for the model.  

<font size = '5'>Please run the project.py to get the result.
