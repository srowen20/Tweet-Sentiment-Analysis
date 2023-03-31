FROM python:3.8.1

# Set up working directory
WORKDIR /

# Copy respective files 
COPY  requirements.txt /requirements.txt
COPY  job-requirements.txt /job-requirements.txt
COPY  beam_preprocessing.py /beam_preprocessing.py

# Install specific requirements
RUN python3 -m pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords


# Set the grpcio environment variables to allow grpcio (async processing)
RUN export GRPC_ENABLE_FORK_SUPPORT=true
RUN export GRPC_POLL_STRATEGY=poll

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "beam_preprocessing.py"]
