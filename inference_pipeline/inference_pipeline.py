from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component, Condition
from google.cloud import aiplatform
from datetime import datetime
import random
import yaml
from typing import NamedTuple

RUN_PIPELINE = True

# pipeline runtime for versioning:

pipeline_runtime = datetime.now().strftime('%Y%m%d%H%M%S')

# import config variables:
with open('kubeflow_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
locals().update(cfg)


@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.8"
)
def get_model_resource_name(project: str,
                  location: str,
                  model_display_name: str
                  ) -> str:
    """
    Extract AutoML model resource name to use as our model
    """

    from google.cloud import aiplatform

    aiplatform.init(project=project,
                    location=location)
    models = aiplatform.Model.list(filter=f"display_name={model_display_name}", order_by='create_time')

    return models[-1].resource_name


@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.8"
)
def get_model_number(model_parent: str
                  ) -> str:
    """
    Extract AutoML model resource name to use as our model
    """

    model_number = model_parent.split('/')[-1]

    return model_number


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
                    bq_processed_table: str,
                    bq_dataset: str,
                    preprocess_data_job_name: str,
                    preprocess_data_image_uri: str,
                    preprocess_data_machine_type: str,
                    service_account: str,
                    output_csv: str
                   ) -> str:
    """
    Runs out beam processing job
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
            '--requirements_file=job-requirements.txt',
            '--is-inference-data=' + str(is_inference_data),
            '--gcs-file=' + gcs_file,
            '--raw-bq-table=' + bq_raw_table,
            '--processed-bq-table=' + bq_processed_table,
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
        
    return bq_processed_table


@component(
    packages_to_install=["google-cloud-aiplatform", "pandas",  "db-dtypes"],
    base_image="python:3.8"
)
def bq_table_to_csv(
                    project: str,
                    bq_dataset: str,
                    bq_processed_table: str,
                    bucket_name: str,
                    processed_file_name: str
                    ) -> str:

    from google.cloud import bigquery
    from google.cloud import storage
    import pandas
    
    client1 = bigquery.Client()

    dataset_ref = bigquery.DatasetReference(project, bq_dataset)
    table_ref = dataset_ref.table(bq_processed_table)
    table = client1.get_table(table_ref)

    df = client1.list_rows(table).to_dataframe()

    client2 = storage.Client()
    bucket = client2.get_bucket(bucket_name)

    bucket.blob(processed_file_name).upload_from_string(df.to_csv(), 'text/csv')
    output_uri = f'gs://{bucket_name}/{processed_file_name}'

    return output_uri


@component(
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage", "pandas"],
    base_image="python:3.8"
)

def get_batch_prediction(
        inference_dataset_source: str,
        source_bucket: str,
        inference_bucket: str,
        project: str,
        location: str,
        model_parent: str) -> bool:
    
    from google.cloud import aiplatform
    from google.cloud import storage
    import time
    import calendar
    import json
    from datetime import date
    import pandas as pd

    ### HELPER FUNCTIONS
    def write_string_to_gcs_txt(string ,file_name, bucket_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(string)

    def upload_blob(source_file_name, destination_blob_name, bucket_name):
      """Uploads a file to the bucket."""
      storage_client = storage.Client()
      bucket = storage_client.get_bucket(bucket_name)
      blob = bucket.blob(destination_blob_name)

      blob.upload_from_filename(source_file_name)

      print('File {} uploaded to {}.'.format(
          source_file_name,
          destination_blob_name))

    def list_blobs(bucket_name , prefix = None):
        """Lists all the blobs in the bucket."""
        # bucket_name = "your-bucket-name"

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name, prefix = prefix)

        # Note: The call returns a response only when the iterator is consumed.
        for blob in blobs:
            print(blob.name)

    def create_batch_prediction_job_sample(
        project: str,
        location: str,
        model_resource_name: str,
        job_display_name: str,
        gcs_source: str,
        gcs_destination: str,
        sync: bool = True,
                            ):

        import pandas as pd
        import google
        from google.cloud import storage
        from google.cloud import aiplatform

        import time
        import calendar
        import json
        from datetime import date

        aiplatform.init(project=project, location=location)

        my_model = aiplatform.Model(model_resource_name)

        batch_prediction_job = my_model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=gcs_source,
            gcs_destination_prefix=gcs_destination,
            sync=sync,
        )

        batch_prediction_job.wait()

        print(batch_prediction_job.display_name)
        print(batch_prediction_job.resource_name)
        print(batch_prediction_job.state)
        
        return batch_prediction_job
    return True


@pipeline(
    name=pipeline_name,
    description='inference pipeline'
)
def inference_pipeline(project: str,
                      location: str,
                      bucket_name: str,                       
                      model_display_name: str,
                      service_location: str,
                      is_inference_data: bool,
                      gcs_file: str,
                      bq_raw_table: str,
                      bq_processed_table: str,
                      bq_dataset: str,
                      preprocess_data_job_name: str,
                      preprocess_data_image_uri: str,
                      preprocess_data_machine_type: str,
                      processed_file_name: str,                      
                      service_account: str,
                      output_csv: str):

    get_model_task = get_model_resource_name(project=project,
                                            location=location,
                                            model_display_name=model_display_name)
    
    #get_model_number_task = get_model_number(model_parent = get_model_task.output)processed_file_name
    
    preprocess_task = preprocess_data(project=project,
                                location=location,
                                is_inference_data=is_inference_data,
                                service_location=service_location,
                                gcs_file=gcs_file,
                                bq_raw_table=bq_raw_table,
                                bq_processed_table=bq_processed_table,
                                bq_dataset=bq_dataset,
                                preprocess_data_job_name=preprocess_data_job_name,
                                preprocess_data_image_uri=preprocess_data_image_uri,
                                preprocess_data_machine_type=preprocess_data_machine_type,
                                service_account=service_account,
                                output_csv = output_csv)
    
    bq_table_to_csv_task = bq_table_to_csv(project=project,
                    bq_dataset=bq_dataset,
                    bq_processed_table = preprocess_task.output,                                           
                    bucket_name = bucket_name,
                    processed_file_name = processed_file_name
                    )

    get_batch_prediction_task =  get_batch_prediction(  inference_dataset_source = bq_table_to_csv_task.output,
                                                        source_bucket = bucket_name,
                                                        inference_bucket = inference_bucket,
                                                        project = project,
                                                        location = location,
                                                        model_parent = get_model_task.output)

if RUN_PIPELINE:
    # Compile pipeline
    pipeline_func = inference_pipeline
    pipeline_filename = pipeline_func.__name__ + '.json'
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    #specify our pipeline parameters for our job:
    PIPELINE_PARAMETERS = {
                            'project': project,
                            'location': location,
                            'bucket_name': bucket_name,
                            'model_display_name': model_display_name,
                            'service_account':service_account,
                            'service_location': service_location,
                            'is_inference_data': is_inference_data,
                            'gcs_file': gcs_file,
                            'bq_raw_table': bq_raw_table,
                            'bq_processed_table': bq_processed_table,
                            'bq_dataset': bq_dataset,
                            'preprocess_data_job_name': preprocess_data_job_name,
                            'preprocess_data_image_uri': preprocess_data_image_uri,
                            'preprocess_data_machine_type': preprocess_data_machine_type,
                            'processed_file_name': processed_file_name,     
                            'output_csv': output_csv
    }

    display_name = f'{pipeline_name}-{pipeline_runtime}'
    job = aiplatform.PipelineJob(display_name=display_name,
                                 template_path=pipeline_filename,
                                 job_id=display_name,
                                 pipeline_root=f'gs://{bucket_name}',
                                 parameter_values=PIPELINE_PARAMETERS,
                                 location = location,enable_caching=False
    )

    job.submit()
