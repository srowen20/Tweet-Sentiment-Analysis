from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component
from google.cloud import aiplatform
from datetime import datetime
import yaml

pipeline_runtime = datetime.now().strftime('%Y%m%d%H%M%S')

with open('inference_config.yaml', 'r') as f:
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
bq_raw_table=cfg['bq_raw_table']
bq_vertex_ai_table = cfg['bq_vertex_ai_table']
dataset_display_name = cfg['dataset_display_name']
ENABLE_CACHING = cfg['enable_caching']
gcs_labelled_csv_file=cfg['gcs_labelled_csv_file']
output_csv=cfg['output_csv']
raw_headers=cfg["raw_headers"]


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
                        raw_headers: str
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
    This function loads an input csv from GCS as a blob and 
    saves the output to BigQuery as a table.
    """
    # Initialise the BigQuery client and download the input csv as a blob
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    filename_start_index = gcs_file.index(bucket_name) + len(bucket_name)
    filename = os.path.basename(gcs_file[filename_start_index:])
    blob = bucket.blob(filename)
    blob = blob.download_as_string()
    
    # Use an OS agnostic codec to decode the blob
    decoded_string = blob.decode('iso-8859-1')
    
    # Create a header pattern from the header string
    header_string = ','.join(raw_headers)
    header_pattern = header_string + '\r\n'     # csv header format exported from windows OS 
    header_pattern_v2 = header_string + '\r'    # csv header format exported from mac OS
    
    # Take a copy of the entire string for restartability
    decoded_string_copy = decoded_string
    
    # Match the header pattern string to the string blob
    match_object = re.match(header_pattern, decoded_string_copy)
    if match_object is None:
        match_object=re.match(header_pattern_v2, decoded_string_copy)
    
    # Remove the header from the string blob, leaving only data entries
    if match_object is not None:
        start, end = match_object.span()
        header = decoded_string_copy[start:end]
        decoded_string_copy = decoded_string_copy[end:]

    # create lists for each row element to be identified and errored tweets (dummy variable)
    training_data_json_items = list()
    errored_tweets = list()
    
    pattern = "(?s)([0-9]{1,9}),([0-9]{5}),(.*?),([0-9]{2}-[0-9]{2}-[0-9]{4})?" # e.g. '14,13254,London UK,01-02-2021'
    
    # Get a list of the start indexes where the above pattern matches elements in the blob
    search = [m.start(0) for m in re.finditer(pattern, decoded_string_copy)]
    
    # Using the start indexes, split the blob into a list of strings, where each string is one row
    split_string = [decoded_string_copy[i:j] for i,j in zip(search, search[1:])]
    
    # Create a regex pattern for one row entry
    full_pattern = "(?s)([0-9]{1,9}),([0-9]{5}),(.*?),([0-9-]*),(.*)"
    header_list = raw_headers.split(',')
    
    # iterate through the list of strings, splitting it into a sub-list by column and parse each item into a dict.
    for item in split_string:
        match_obj = re.match(full_pattern, item)
        if match_obj is None:
            errored_tweets.append(item)
        else: 
            training_data_json_items.append(
                {"UserName":match_obj.group(1),
                 "ScreenName": match_obj.group(2),
                 "Location": match_obj.group(3),
                 "TweetAt": match_obj.group(4),
                 "OriginalTweet": match_obj.group(5),
                })

    # Convert the list of row dicts to a dataframe and add a UUID column
    training_data_df = pd.DataFrame(training_data_json_items)
    training_data_df['uuid'] = [str(uuid.uuid4()) for _ in range(len(training_data_df.index))]
    
    # Define BQ client object and write the dataframe to BQ
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
                      gcs_labelled_csv_file: str,
                      bucket_name: str,
                      output_csv: str,
                      raw_headers: str,
                     ):
    
    load_raw_data = load_gcs_blob_to_bq(project=project, 
                                        bq_raw_table=bq_raw_table,
                                        gcs_file=gcs_file,
                                        bucket_name=bucket_name,
                                        bq_dataset=bq_dataset,
                                        location=location,
                                        raw_headers=raw_headers)
    
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
                       "gcs_labelled_csv_file": gcs_labelled_csv_file,
                       "bucket_name": bucket_name,
                       "output_csv": output_csv,
                       "bq_vertex_ai_table": bq_vertex_ai_table,
                       "raw_headers": raw_headers}

display_name = f'{pipeline_name}--{pipeline_runtime}'
job = aiplatform.PipelineJob(display_name=display_name,
                             template_path=pipeline_filename,
                             job_id=display_name,
                             pipeline_root=f'gs://{bucket_name}',
                             parameter_values=PIPELINE_PARAMETERS,
                             enable_caching=ENABLE_CACHING,
                             location=location)

job.submit(service_account=service_account)
