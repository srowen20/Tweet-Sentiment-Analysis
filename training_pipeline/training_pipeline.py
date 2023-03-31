from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component
from google.cloud import aiplatform
from datetime import datetime
import yaml

pipeline_runtime = datetime.now().strftime('%Y%m%d%H%M%S')

with open('config.yaml', 'r') as f:
    cfg = yaml.load(f)

pipeline_name = cfg['pipeline_name']
project = cfg['project']
location = cfg['location']
service_account = cfg['service_account']
is_inference_data = cfg['is_inference_data']
gcs_file = cfg['gcs_file']
service_location=cfg['service_location']
preprocess_data_job_name=cfg['preprocess_data_job_name']
preprocess_data_image_uri=cfg['preprocess_data_image_uri']
preprocess_data_machine_type=cfg['preprocess_data_machine_type']
bucket_name = cfg['bucket_name']
bq_dataset = cfg['bq_dataset']
model_type = cfg['model_type']
model_display_name = cfg['model_display_name']
bq_raw_table=cfg['bq_raw_table']
bq_vertex_ai_table = cfg['bq_vertex_ai_table']
dataset_display_name = cfg['dataset_display_name']
ENABLE_CACHING = cfg['enable_caching']
gcs_labelled_csv_file=cfg['gcs_labelled_csv_file']
training_bucket=cfg['training_bucket']
output_csv=cfg['output_csv']
sentiment_max=cfg["sentiment_max"]
raw_headers=cfg["raw_headers"]
raw_sentiments=cfg["raw_sentiments"]

@component(
    packages_to_install=["chardet", "pandas", "google-cloud-bigquery",
                         "google-cloud-storage", "pyarrow"],
    base_image="python:3.8"
)
def load_gcs_blob_to_bq(project: str,
                      bq_raw_table: str,
                      gcs_file: str,
                      bucket_name: str,
                      bq_dataset: str,
                      location: str,
                      raw_sentiments: list,
                      raw_headers: str,
                      is_inference_data: bool
                     ) -> bool:
    from google.cloud import storage
    from google.cloud import bigquery
    import pandas as pd
    import os
    import chardet
    import re
    from typing import List
    import pandas as pd
    import uuid
    
    """
    Loads the csv as a blob and saves to bigquery as a table.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    filename_start_index = gcs_file.index(bucket_name) + len(bucket_name)
    filename = os.path.basename(gcs_file[filename_start_index:])
    blob = bucket.blob(filename)
    blob = blob.download_as_string()
    
    # Use an OS agnostic codec to decode the blob into a long string
    decoded_string = blob.decode('iso-8859-1')
    
    # Create a list of sentiments, adding underscores either side to differentiate from tweet content
    sentiments = list(map((lambda x: '_' + x.replace(' ', '-') + '_'), raw_sentiments))
    header_pattern = raw_headers + '\r\n'
    
    # Replace all labels with custom labels. It helps further processing
    for i in range(len(sentiments)):
        decoded_string = decoded_string.replace(raw_sentiments[i] + '\r\n', sentiments[i]) 
    
    # Take a copy of the entire string for restartability
    decoded_string_copy = decoded_string
    
    # Match the header pattern and remove from the string
    match_object = re.match(header_pattern, decoded_string_copy)
    if match_object is not None:
        start, end = match_object.span()
        header = decoded_string_copy[start:end]
        decoded_string_copy = decoded_string_copy[end:]
            
    def get_tweet_string(x: str,
                     sentiments: List =sentiments):
        """
        The function locates the locations of specified sentiment categories from a string.
        The string is then stripped at the closest category location and then return both the 
        striped portion and remaining string. 

        returns tweeet (str), rest of the string blob (str)"""
        positions = {}

        for i in sentiments:
            pos = x.find(i)
            if pos!= -1:
                positions[i]=pos

        if bool(positions):
            positions_ordered = dict(sorted(positions.items(), key=lambda item: item[1]))
            first, *rest = positions_ordered
            return x[:positions_ordered[first]+len(first)], x[positions_ordered[first]+len(first):]
        else:
            return '',''
        
    # Initialise outputs
    training_items = list()
    
    # Execute the above function iteratively, until the end of the main string. 
    # Each time append the tweet string to the output list
    while len(decoded_string_copy) > 0:
        tweet, decoded_string_copy = get_tweet_string(decoded_string_copy)
        training_items.append(tweet)

    # regex pattern to parse each item in the training list
    pattern = "(?s)([0-9]{1,9}),([0-9]{1,9}),(.*?),([0-9-]*),(.*),(_Positive_|_Negative_|_Neutral_|_Extremely-Positive_|_Extremely-Negative_)"
    training_data_json_items = list()
    errored_tweets = list()
    
    # Select the dict to capture the counts of categories during parsing
    sentiment_counts = {}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = 0
    header_list = raw_headers.split(',')
    
    # iterate through the training items list and parse each item into a dict. Also capture the counts.
    for item in training_items:
        match_obj = re.match(pattern, item)
        if match_obj is None:
            errored_tweets.append(item)
        else: 
            training_data_json_items.append(
                {"UserName":match_obj.group(1),
                 "ScreenName": match_obj.group(2),
                 "Location": match_obj.group(3),
                 "TweetAt": match_obj.group(4),
                 "OriginalTweet": match_obj.group(5),
                 "Sentiment": match_obj.group(6)
                })

    # Load the data to bigquery/cloud storage
    training_data_df = pd.DataFrame(training_data_json_items)
    training_data_df['Sentiment'] = training_data_df['Sentiment'].str.replace('_', '').replace('-', ' ')
    training_data_df['uuid'] = [str(uuid.uuid4()) for _ in range(len(training_data_df.index))]
    client = bigquery.Client(project=project)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )
    write_table = f"{project}.{bq_dataset}.{bq_raw_table}"
    job = client.load_table_from_dataframe(
        training_data_df, write_table, job_config=job_config
    )
    
    return True
        
@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.8"
)
def preprocess_data(project: str,
                    location: str,
                    is_inference_data: bool,
                    service_location: str,
                    gcs_file: str,
                    bq_raw_table: str,
                    bq_vertex_ai_table: str,
                    bq_dataset: str,
                    preprocess_data_job_name: str,
                    preprocess_data_image_uri: str,
                    preprocess_data_machine_type: str,
                    service_account: str,
                    output_csv: str,
                    dummy_var: bool
                   ) -> bool:
    """
    Runs our beam processing job
    """
    
    from google.cloud import aiplatform as ai
    from time import sleep
    from datetime import datetime
    
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    client = ai.gapic.JobServiceClient(client_options=client_options)
    
    args = ['--project=' + project,
            '--region=' + location,
            '--runner=DataflowRunner',
            '--temp_location=' + service_location,
            '--requirements_file=requirements.txt',
            '--is-inference-data=' + str(is_inference_data),
            '--gcs-file=' + gcs_file,
            '--raw-bq-table=' + bq_raw_table,
            '--processed-bq-table=' + bq_vertex_ai_table,
            '--bq-dataset=' + bq_dataset,
            '--gcp-project=' + project,
            '--gcs-output-csv=' + output_csv,
           ]
    
    custom_job = {
        "display_name": preprocess_data_job_name + '-' + datetime.now().strftime("%Y%m%d%H%M%S"),
        "job_spec": {
            "service_account": service_account,
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": preprocess_data_machine_type,
                    },
                    "replica_count": 1,
                    "disk_spec": {
                        "boot_disk_type": "pd-ssd",
                        "boot_disk_size_gb": 1000
                    },
                    "container_spec": {
                        "image_uri": preprocess_data_image_uri,
                        "command": [],
                        "args": args
                    },
                }
            ]
        },
    }
    
    parent = f"projects/{project}/locations/{location}"
    
    try:
        response = client.create_custom_job(parent=parent, custom_job=custom_job)
    except Exception as e:
        print(f"Error encountered while submitting custom job. Error {e}")
        raise Exception("Unable to submit raw data load job error")
    else:
        print(f"Submitted custom job")
        
    end_job = False
    
    # Wait for the job to reach end state
    while not end_job:
        get_response = client.get_custom_job(name=response.name)
        
        print(f"Custom job state : {get_response.state.name}")
        
        if get_response.state.name in ['JOB_STATE_SUCCEEDED',
                                       'JOB_STATE_FAILED',
                                       'JOB_STATE_CANCELLED',
                                       'JobState.JOB_STATE_PAUSED',
                                       'JobState.JOB_STATE_EXPIRED']:
            end_job = True
        else:
            print("Waiting for the job to complete")
            sleep(120)
    
    if get_response.state.name == 'JOB_STATE_SUCCEEDED':
        print("Custom job completed successfully")
    else:
        print(f"Custom job is not successful. Job state {get_response.state.name}")
        raise Exception("Custom job is not successful")
        
    return True


@component(
    packages_to_install=['google-cloud-bigquery', 'pandas',
                         'scikit-learn', 'db-dtypes', 'fsspec',
                        'google-cloud-storage'],
    base_image="python:3.8"
)
def create_training_inputs(project: str,
                          bq_dataset: str,
                          bq_vertex_ai_table: str,
                          gcs_labelled_csv_file: str,
                          training_bucket: str,
                          dummy_var: bool,
                          sentiment_max: int) -> bool:
    
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from google.cloud import storage
    
    client = bigquery.Client(project=project)
    sql_query = f"""
        SELECT * 
        FROM `{project}.{bq_dataset}.{bq_vertex_ai_table}`
    """
    
    processed_df = client.query(sql_query).to_dataframe()
    
    def stratified_split(df):
        X = df
        y = df['Sentiment']
        X_train, X_testval, y_train, y_testval = train_test_split(X, y,
                                                                  stratify=y,
                                                                  test_size=0.2,
                                                                  random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval,
                                                        stratify=y_testval,
                                                        test_size=0.5,
                                                        random_state=42)
        X_train['Split'] = 'training'
        X_test['Split'] = 'test'
        X_val['Split'] = 'validation'
        
        data = pd.DataFrame(columns=['Split', 'CleanedTweet', 'Sentiment', 'uuid'])
        X = X_train.append(X_test)
        X = X.append(X_val)
        
        data['Split'] = X['Split']
        data['CleanedTweet'] = X['CleanedTweet']
        data['Sentiment'] = X['Sentiment']
        data['MaxSentiment'] = sentiment_max
        data['uuid'] = X['uuid']
        return data
    
    bq_df = stratified_split(processed_df)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )
    
    write_table = f"{project}.{bq_dataset}.{bq_vertex_ai_table}"
    job = client.load_table_from_dataframe(
        bq_df, write_table, job_config=job_config
    )
    
    def upload_to_bucket(training_bucket, gcs_labelled_csv_file, df):
        csv_df = df.drop(columns='uuid')
        bucket = storage.Client().get_bucket(training_bucket)
        bucket.blob(gcs_labelled_csv_file).upload_from_string(csv_df.to_csv(index=False, header=False), 'text/csv') 
    
    upload_to_bucket(training_bucket, gcs_labelled_csv_file, bq_df)
    
    job.result()
    
    return True

@component(
    packages_to_install=['google-api-core', 'google-cloud-aiplatform'],
    base_image="python:3.8"
)
def create_vertex_ai_dataset(
    display_name: str,
    project: str,
    location: str,
    bq_dataset: str,
    bq_vertex_ai_table: str,
    gcs_labelled_csv_file: str,
    training_bucket: str,
    dummy_var: bool) -> str:
    
    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=location)
    
    import_schema_uri = aiplatform.schema.dataset.ioformat.text.sentiment
    
    dataset = aiplatform.TextDataset.create(
        display_name=display_name,
        project=project,
        location=location,
        gcs_source=f'gs://{training_bucket}/{gcs_labelled_csv_file}',
        import_schema_uri=import_schema_uri
    )
    
    dataset.wait()
    
    print(f'\tDataset: "{dataset.display_name}"')
    print(f'\tname: "{dataset.resource_name}"')
    
    return dataset.resource_name

@component(
    packages_to_install=['google-api-core', 'google-cloud-aiplatform'],
    base_image="python:3.8"
)
def train_model(
    project: str,
    display_name: str,
    dataset_id: str,
    location: str,
    model_display_name: str = None,
    model_type: str = 'sentiment',
    sentiment_max: int = 2,
    sync: bool = True) -> str:
    
    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=location)
    
    training_job = aiplatform.AutoMLTextTrainingJob(
        display_name=display_name, 
        prediction_type=model_type,
        sentiment_max=sentiment_max,
    )
    
    text_dataset = aiplatform.TextDataset(dataset_name=dataset_id)
    
    model = training_job.run(
        dataset=text_dataset,
        model_display_name=model_display_name,
        training_filter_split="labels.aiplatform.googleapis.com/ml_use=training",
        validation_filter_split="labels.aiplatform.googleapis.com/ml_use=validation",
        test_filter_split="labels.aiplatform.googleapis.com/ml_use=test",
        sync=sync,
    )
    
    model.wait()

    print(model.display_name)
    print(model.resource_name)
    print(model.uri)
    return model.resource_name

@component(
    packages_to_install=['google-api-core',
                         'google-cloud-aiplatform',
                         'google-cloud-bigquery',
                         'pandas',
                         'pyarrow', 'db-dtypes'],
    base_image="python:3.8"
)
def evaluate_model(model: str,
                   project: str,
                   location: str,
                   bq_dataset: str,
                   bq_vertex_ai_table: str) -> bool:
    
    from google.cloud import aiplatform, bigquery
    import pandas as pd
    import math
    from itertools import islice

    def get_model_performance_data(model_id, overall_eval, slices=False):

        eval_df = pd.DataFrame(columns=['label',
                                        'model_id',
                                        'no_test_items',
                                        'no_train_items',
                                        'no_val_items',
                                        'f1Score',
                                        'precision',
                                        'recall'])

        

        try: 
            label = str(overall_eval.slice_).split()[-1].replace('"', '')
        except AttributeError:
            label = 'overall'

        client = bigquery.Client(project=project)

        bq_sql_query = f"""
        SELECT * FROM `{project}.{bq_dataset}.{bq_vertex_ai_table}`"""

        split = client.query(bq_sql_query)
        split.result()
        split_df = split.to_dataframe()

        training_count_df = split_df.groupby(["Split"])["Split"].count()

        no_test_items = training_count_df['test']
        no_train_items = training_count_df['training']
        no_val_items = training_count_df['validation']
        eval_df = eval_df.append(pd.Series([label,
                                            model_id,
                                            no_test_items,
                                            no_train_items,
                                            no_val_items,
                                            overall_eval.metrics['f1Score'],
                                            overall_eval.metrics['precision'],
                                            overall_eval.metrics['recall']], index=eval_df.columns), ignore_index=True)

        if slices == False:
            cm_df = pd.DataFrame(columns=['model_id',
                                          'sentiment_actuals',
                                          'negative_pred',
                                          'neutral_pred',
                                          'positive_pred'])
            pct_cm_df = pd.DataFrame(columns=['model_id',
                                              'sentiment_actuals',
                                              'negative_pred',
                                              'neutral_pred',
                                              'positive_pred'])
            cm = overall_eval.metrics['confusionMatrix']['rows']
            sentiments = ['negative', 'neutral', 'positive']

            pct_cm = convert_cm_to_percentage(cm)

            for i in range(len(cm)):
                cm_df = cm_df.append(pd.Series([model_id, sentiments[i], cm[i][0], cm[i][1], cm[i][2]], index=cm_df.columns), ignore_index=True)
                pct_cm_df = pct_cm_df.append(pd.Series([model_id, sentiments[i], pct_cm[i][0], pct_cm[i][1], pct_cm[i][2]], index=cm_df.columns), ignore_index=True)

            return eval_df, cm_df, pct_cm_df
        else:
            return eval_df

    def convert_cm_to_percentage(cm):
        confusion_percentage_accuracies = []
        for i in cm:
            for j in i:
                confusion_percentage_accuracies.append(j/int(sum(i)))

        cm_size = int(math.sqrt(len(confusion_percentage_accuracies)))
        length_to_split = [cm_size] * cm_size
        iter_item = iter(confusion_percentage_accuracies)
        confusion_percentage_array = [list(islice(iter_item, elem)) for elem in length_to_split]

        return confusion_percentage_array

    def get_model_evaluations(
        model: str,
        location: str,
        model_id: str,
        include_slices=False
    ):

        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}

        model_client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
        list_eval = model_client.list_model_evaluations(parent=model)
        for evaluation in list_eval:
            eval_name = evaluation.name

        overall_eval = model_client.get_model_evaluation(name=eval_name)

        eval_df, cm_df, pct_cm_df = get_model_performance_data(model_id, overall_eval)

        if include_slices:
            slice_names = []
            list_slices = model_client.list_model_evaluation_slices(parent=eval_name)
            for data in list_slices:
                slice_names.append(data.name)
            
            for slice_name in slice_names:
                eval_slice = model_client.get_model_evaluation_slice(name=slice_name)
                temp_eval_df = get_model_performance_data(model_id, eval_slice, slices=True)
                eval_df = eval_df.append(temp_eval_df, ignore_index=True)


        return eval_df, cm_df, pct_cm_df

    def load_dataframe_to_bigquery(df, table_id):

        client = bigquery.Client(project=project)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result() # Wait for the job to complete

        table = client.get_table(table_id)
        print(f'Loaded {table.num_rows} rows and {len(table.schema)} columns to {table_id}')

    model_id = model.split('/')[-1]    
    eval_df, cm_df, pct_cm_df = get_model_evaluations(model, location, model_id=model_id, include_slices=True)


    load_dataframe_to_bigquery(eval_df, table_id=f'{project}.{bq_dataset}.model_evaluations')
    load_dataframe_to_bigquery(cm_df, table_id=f'{project}.{bq_dataset}.confusion_matrix')
    load_dataframe_to_bigquery(pct_cm_df, table_id=f'{project}.{bq_dataset}.confusion_matrix_percentage')
    
    return True

@pipeline(
    name=pipeline_name,
    description='training pipeline'
)
def training_pipeline(project:str,
                      location: str,
                      service_location: str,
                      is_inference_data: bool,
                      gcs_file: str,
                      bq_raw_table: str,
                      bq_dataset: str,
                      preprocess_data_job_name: str,
                      preprocess_data_image_uri: str,
                      preprocess_data_machine_type: str,
                      service_account: str,
                      bq_vertex_ai_table: str,
                      dataset_display_name: str,
                      model_display_name: str,
                      model_type: str,
                      gcs_labelled_csv_file: str,
                      training_bucket: str,
                      output_csv: str,
                      sentiment_max: int,
                      raw_sentiments: list,
                      raw_headers: str
                     ):
    
    load_raw_data = load_gcs_blob_to_bq(project=project, 
                                      bq_raw_table=bq_raw_table,
                                      gcs_file=gcs_file,
                                      bucket_name=bucket_name,
                                      bq_dataset=bq_dataset,
                                      location=location,
                                      raw_sentiments=raw_sentiments,
                                      raw_headers=raw_headers,
                                      is_inference_data=is_inference_data)
    
    preprocess_task = preprocess_data(project=project, 
                                      location=location,
                                      service_location=service_location,
                                      is_inference_data=is_inference_data,
                                      gcs_file=gcs_file,
                                      bq_raw_table=bq_raw_table,
                                      bq_vertex_ai_table=bq_vertex_ai_table,
                                      bq_dataset=bq_dataset,
                                      preprocess_data_job_name=preprocess_data_job_name,
                                      preprocess_data_image_uri=preprocess_data_image_uri,
                                      preprocess_data_machine_type=preprocess_data_machine_type,
                                      service_account=service_account,
                                      output_csv=output_csv,
                                      dummy_var=load_raw_data.output)
    
    training_inputs_task = create_training_inputs(project=project,
                                                  bq_dataset=bq_dataset,
                                                  bq_vertex_ai_table=bq_vertex_ai_table,
                                                  gcs_labelled_csv_file=gcs_labelled_csv_file,
                                                  training_bucket=training_bucket,
                                                  sentiment_max=sentiment_max,
                                                  dummy_var=preprocess_task.output)
    
    dataset_task = create_vertex_ai_dataset(display_name=dataset_display_name,
                                            project=project,
                                            location=location,
                                            bq_dataset=bq_dataset,
                                            bq_vertex_ai_table=bq_vertex_ai_table,
                                            gcs_labelled_csv_file=gcs_labelled_csv_file,
                                            training_bucket=training_bucket,
                                            dummy_var=training_inputs_task.output)
    
    train_model_task = train_model(project=project,
                                   display_name=model_display_name,
                                   dataset_id=dataset_task.output,
                                   location=location,
                                   model_display_name=model_display_name,
                                   model_type=model_type,
                                   sentiment_max=sentiment_max)
    
    evaluate_model_task = evaluate_model(model=train_model_task.output,
                                         project=project,
                                         location=location,
                                         bq_dataset=bq_dataset,
                                         bq_vertex_ai_table=bq_vertex_ai_table)

# Compile pipeline
pipeline_func = training_pipeline
pipeline_filename = pipeline_func.__name__ + '.json'
compiler.Compiler().compile(pipeline_func, pipeline_filename)

# Specify our pipeline parameters for our job:
PIPELINE_PARAMETERS = {"project": project,
                       "location": location,
                       "service_account":service_account,
                       "service_location": service_location,
                       "is_inference_data": is_inference_data,
                       "gcs_file": gcs_file,
                       "bq_raw_table": bq_raw_table,
                       "bq_dataset": bq_dataset,
                       "preprocess_data_job_name": preprocess_data_job_name,
                       "preprocess_data_image_uri": preprocess_data_image_uri,
                       "preprocess_data_machine_type": preprocess_data_machine_type,
                       "dataset_display_name": dataset_display_name,
                       "bq_vertex_ai_table": bq_vertex_ai_table,
                       "model_display_name": model_display_name,
                       "model_type": model_type,
                      "gcs_labelled_csv_file": gcs_labelled_csv_file,
                      "training_bucket": training_bucket,
                      "output_csv": output_csv,
                      "bq_vertex_ai_table": bq_vertex_ai_table,
                      "sentiment_max": sentiment_max,
                      "raw_headers": raw_headers,
                      "raw_sentiments": raw_sentiments}

display_name = f'{pipeline_name}--{pipeline_runtime}'
job = aiplatform.PipelineJob(display_name=display_name,
                             template_path=pipeline_filename,
                             job_id=display_name,
                             pipeline_root=f'gs://{bucket_name}',
                             parameter_values=PIPELINE_PARAMETERS,
                             enable_caching=ENABLE_CACHING,
                             location=location)

job.submit(service_account=service_account)
