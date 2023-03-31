# Covid-Tweets Sentiment Analysis

This project demonstrates processing a Covid Tweet dataset from Kaggle and training and AutoML Model to predict the Sentiment of unlabelled tweets. 

The data was from: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv 

It is written in such a way that it could predict the sentiment of any tweet fed in (not necessarily covid tweets). 

If running locally, run `gcloud auth application-default login` to authorise your account with the Google Cloud Platform. 

## Brief

Create a reusable Text Sentiment Analysis Vertex AI pipeline. 

- Use Dataflow to load data to BigQuery, retaining raw and processed data
- Carry out EDA on the dataset
- Compare AutoML sentiment analysis, Google NLP API and other open-source options such as SpaCy
- Build a Kubeflow pipeline
- Use Looker to visualise results & insights (still to be completed)

## Required outputs

- EDA notebook which can be reused for any dataset
- Modelling notebooks which can be reused for any dataset
- Different models tested and compared
- Training pipeline that can be applied to any text classification problem
- Inference pipeline

## Preprocessing

Preprocessing was done in Apache Beam using a DataFlow Runner. Some preprocessing steps included removing stopwords, expanding contractions, and removing hashtags, mentions, punctuation and URLs. 

An additional step was added to the Kubeflow pipeline before the Dataflow job was kicked off, which loaded the CSV file as a blob. This avoided errors from loading the file as a CSV and the loading method assuming new lines within a tweet indicated a new row in the dataset. The blob could then be searched for Regex patterns within it to separate each row. There were some encoding errors in the `Corona_NLP_train.csv` file which were fixed by running a ParDo function in the Apache Beam job. 

## Training Pipeline

Kick off the training pipeline by:
1. Editing the `training_pipeline/config.yaml` file with the correct bucket, project and more
2. Run `training_pipeline.py`
3. The job progress can be tracked in Vertex AI -> Pipelines

The model is likely to take a couple of hours to train. Once trained, it can be found in the Vertex AI Model Registry. 

## Inference Pipeline

1. Edit the file `inference_pipeline/kubeflow_config.yaml`
2. Run `inference_pipeline.py`

## Directories

- `api_predictions` - contains code which calls the Spacy and Google APIs to make predictions on tweets
    - `APIs_spacy_google.ipynb` calls the APIs for predictions on the sentiment, then evaluates the results.
- `data` - the train and test `.csv` files downloaded from Kaggle
- `exploratory_analysis` - contains one EDA file with statistical analysis on the raw dataset
- `preprocessing` - contains the Docker file, apache beam code and requirements file for containerising the code
- `training_pipeline` - kubeflow pipeline to train an AutoML model
- `inference_pipeline` - kubeflow pipeline to run inferences on unlabelled data

