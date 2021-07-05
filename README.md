# Fake_Job_Prediction
This is a model to predict the fake job post. Dataset is job_train.csv.  
Installed packages: pandas, numpy, sklearn.  
  
In the dataset, there are 7 features and 1 label. 
Features are title(the job title of this post - ex: Web Designer), location(the working place of this post - ex: US, TX, USA Southwest), description(the detail of this job description), requirement(the criteria to fulfill the basic skills in this job), telecomuting(do the employer leave the phone number, ex: 1 or 0), has_company_logo(does this company have logo? ex: 1 or 0), has_questions(do they leave some questions on this post, ex: 1 or 0). 
Label is fradulent(Is this post a real job post or it is fake? ex: 1 or 0).   
In this model, TfidVectorizer was implemented to turn the effective features from words into vectors, and SGDClassifier was implemented to train this model with the Grid Search to find the best parameters for the model.   
  
Please run the project.py to get the result.  
