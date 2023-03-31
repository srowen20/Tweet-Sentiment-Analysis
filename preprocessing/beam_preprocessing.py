import argparse
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.dataframe.io import read_csv

import pandas as pd
import re
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
import string
import numpy as np


def row_to_dict(element, is_inference_data):
    """Convert each row to a dictionary, with keys being column names.
    """
    if is_inference_data == 'true':
        output = {'UserName': element[0],
                'ScreenName': element[1],
                'Location': element[2],
                'TweetAt': element[3],
                'OriginalTweet': element[4]
        }

    else:
        output = {'UserName': element[0],
                'ScreenName': element[1],
                'Location': element[2],
                'TweetAt': element[3],
                'OriginalTweet': element[4],
                'Sentiment': element[5]
        }

    return output

class ReplaceEncodingErrors(beam.DoFn):
    def process(self, element):
        rep = {"&amp;": " and ", 
               "&lt;": " < ",
               "&gt;": " > ",
               " s ": "'s ", 
               " amp ": " and "}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], element['OriginalTweet'])

        indexes = []
        for m in re.finditer("â", text):
            indexes.append(m.start())
        indexes.reverse()
        for i in indexes:
            text = text[:i] + "'" + text[i+2:]
        element['OriginalTweet'] = text
        yield element

class EncodeDecode(beam.DoFn):
    def process(self, element):
        """
        Removes errors left in from decoding e.g. '\r\r\n' """
        text_encode = element['OriginalTweet'].encode(encoding="ascii", errors="ignore")
        text_decode = text_encode.decode()
        element['OriginalTweet'] = " ".join([word for word in text_decode.split()])
        yield element

class LowerCase(beam.DoFn):
    def process(self, element):
        element['CleanedTweet'] = element['OriginalTweet'].lower()
        yield element

class ReturnCleanOriginalTweet(beam.DoFn):
    def process(self, element):
        element['OriginalTweet'] = element['OriginalTweet'].capitalize()
        yield element

class ExpandContractions(beam.DoFn):
    def process(self, element):
        import contractions
        expanded_words = []
        expanded_words.append(contractions.fix(element['CleanedTweet']))
        element['CleanedTweet'] = ' '.join(expanded_words)
        yield element

class RemoveHashtags(beam.DoFn):
    def process(self, element):
        element['CleanedTweet'] = re.sub("\#[A-Za-z0-9_]+", "", element['CleanedTweet'])
        yield element

class ReplaceSwearwords(beam.DoFn):
    def process(self, element):
        text = re.sub("[a-z]+\*+[a-z]+", "swearword", element['CleanedTweet'])
        swear = {" f*** ": "swearword",
                " f***** ": "swearword",
                " f****** ": "swearword"
        }
        swear = dict((re.escape(k), v) for k, v in swear.items())
        pattern = re.compile("|".join(swear.keys()))
        element['CleanedTweet'] = pattern.sub(lambda m: swear[re.escape(m.group(0))], text)
        yield element


class ReplaceCommonSlang(beam.DoFn):
    def process(self, element):
        slang = {" er ": " emergency room ",
                " omg ": " oh my god ",
                " fyi ": " for your information ",
                " y ": " why ",
                " btwn ": " between ",
                " jk ": " joke ",
                " dm ": " direct message "
            }
        slang = dict((re.escape(k), v) for k, v in slang.items())
        pattern = re.compile("|".join(slang.keys()))
        element['CleanedTweet'] = pattern.sub(lambda m: slang[re.escape(m.group(0))], element['CleanedTweet'])
        yield element

class RemoveUrls(beam.DoFn):
    def process(self, element):
        element['CleanedTweet'] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", element['CleanedTweet'])
        yield element

class RemoveTags(beam.DoFn):
    def process(self, element):
        char = '@'
        element['CleanedTweet'] = ' '.join(filter(lambda word: not word.startswith(char), element['CleanedTweet'].split()))
        yield element

class RemovePunctuation(beam.DoFn):
    def process(self, element):
        text = element['CleanedTweet'].replace("/", " ")
        element['CleanedTweet'] = text.translate(str.maketrans('', '', string.punctuation))
        yield element

class RemoveNumbers(beam.DoFn):
    def process(self, element):
        element['CleanedTweet'] = element['CleanedTweet'].translate(str.maketrans('', '', string.digits))
        yield element

class Lemmatize(beam.DoFn):
    def process(self, element):
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        text = ' '.join(lemmatizer.lemmatize(word) for word in element['CleanedTweet'].split())
        errors = {" ha ": " has ",
                " wa ": " was "}
        errors = dict((re.escape(k), v) for k, v in errors.items())
        pattern = re.compile("|".join(errors.keys()))
        text = pattern.sub(lambda m: errors[re.escape(m.group(0))], text)
        text = re.sub(r'^wa ', 'was ', text)
        text = re.sub(r'^ha ', 'has ', text)
        text = re.sub(r' wa$', 'was ', text)
        element['CleanedTweet'] = re.sub(r' ha$', 'has ', text)
        yield element

class RemoveStopwords(beam.DoFn):
    def process(self, element):
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stop = set(stopwords.words('english'))
        element['CleanedTweet'] = ' '.join([word for word in element['CleanedTweet'].split() if word not in (stop)])
        yield element

class ReplaceSentimentWithInt(beam.DoFn):
    def process(self, element):
        sentiment_map = {"extremely negative": 0,
                                    "negative": 0,
                                    "neutral": 1,
                                    "extremely positive": 2,
                                    "positive": 2}
        element['Sentiment'] = element['Sentiment'].lower()
        element['Sentiment'] = sentiment_map[element['Sentiment']]
        yield element

class SelectColumns(beam.DoFn):
    
    def __init__(self, keys):
        beam.DoFn.__init__(self)
        self.keys=keys
        
    def process(self, element):
        element = {key: element[key] for key in self.keys}
        yield element

def to_csv_row(element):
    return ','.join(str(v) for _, v in element.items())

def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the preprocessing pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-inference-data', dest='is_inference_data', help='Is this data for inference (rather than training)?')
    parser.add_argument('--gcs-file', dest='gcs_file', help='Input file from GCS to process.')
    parser.add_argument('--raw-bq-table', dest='raw_bq_table', required=True, help='BigQuery raw inputs table.')
    parser.add_argument('--processed-bq-table', dest='processed_bq_table', required=True, help='BigQuery processed inputs table.')
    parser.add_argument('--bq-dataset', dest='bq_dataset', required=True, help='BigQuery dataset.')
    parser.add_argument('--gcp-project', dest='gcp_project', required=True, help='GCP project.')
    parser.add_argument('--gcs-output-csv', dest='output_csv', help='Output file location and name as csv (without extension)')

    known_args, pipeline_args = parser.parse_known_args(argv)

    is_inference_data = str(known_args.is_inference_data).lower()

    # Only have the target column 'Sentiment' in training data, not inference:
    if is_inference_data == 'true':
        raw_column_names = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet']
        raw_schema = 'UserName:INT64, ScreenName:INT64, Location:STRING, TweetAt:STRING, OriginalTweet:STRING'
        processed_column_names = ['OriginalTweet', 'CleanedTweet']
        processed_schema='OriginalTweet:STRING, CleanedTweet:STRING'
    else:
        raw_column_names = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'Sentiment']
        raw_schema = 'UserName:INT64, ScreenName:INT64, Location:STRING, TweetAt:STRING, OriginalTweet:STRING, Sentiment:STRING'
        processed_column_names = ['CleanedTweet', 'Sentiment']
        processed_schema = 'CleanedTweet:STRING, Sentiment:INT64'

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        raw_dataframe = pipeline | 'Read Inputs' >> read_csv(known_args.gcs_file,
                                       encoding='latin1',
                                       delimiter=',',
                                       header=0)
        
        raw_dataframe = beam.dataframe.convert.to_pcollection(raw_dataframe)
        
        raw_data_bq = (
            raw_dataframe 
            | 'Split Rows To Columns' >> beam.Map(row_to_dict,
                                                            is_inference_data=is_inference_data)
        )
        
        # Save the raw data to BigQuery
        write_raw = (
            raw_data_bq 
            | 'Write Raw' >> beam.io.WriteToBigQuery(
                table=known_args.raw_bq_table,
                dataset=known_args.bq_dataset,
                project=known_args.gcp_project,
                schema=raw_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
    

    with beam.Pipeline(options=pipeline_options) as p:
        # Read the raw data file
        raw_df = p | 'Read Inputs' >> read_csv(known_args.gcs_file,
                                       encoding='latin1',
                                       delimiter=',',
                                       header=0)

        raw_df = beam.dataframe.convert.to_pcollection(raw_df)
        
        # format our data into a BigQuery happy format:
        raw_data = (
            raw_df 
            | 'Split Rows To Columns' >> beam.Map(row_to_dict,
                                                            is_inference_data=is_inference_data)
        )
        
        
        # Clean the tweets, going through each function            
        processed_data = (
            raw_data
            | "fix encoding errors" >> beam.ParDo(ReplaceEncodingErrors())
            | beam.ParDo(EncodeDecode())
            | 'Create new CleanedTweet column & convert to lower' >> beam.ParDo(LowerCase())
            | 'Expand Contractions' >> beam.ParDo(ExpandContractions())
            | 'Remove Hashtags' >> beam.ParDo(RemoveHashtags())
            | 'Replace swearwords with placeholders' >> beam.ParDo(ReplaceSwearwords())
            | 'Replace common abbreviations' >> beam.ParDo(ReplaceCommonSlang())
            | 'Remove URLs' >> beam.ParDo(RemoveUrls())
            | 'Remove Tags' >> beam.ParDo(RemoveTags())
            | 'Remove punctuation' >> beam.ParDo(RemovePunctuation())
            | 'Remove numbers' >> beam.ParDo(RemoveNumbers())
            | 'Lemmatize' >> beam.ParDo(Lemmatize())
            | 'Remove stopwords' >> beam.ParDo(RemoveStopwords())
        )
        
        # If not for inference, convert sentiment to an integer and select the required columns
        if is_inference_data == 'false':
            processed = (
                processed_data
                | 'Convert sentiment to integers' >> beam.ParDo(ReplaceSentimentWithInt())
                | 'Select required columns' >> beam.ParDo(SelectColumns(keys=processed_column_names))
            )
        else:
            processed = (
                processed_data
                | 'Select required columns' >> beam.ParDo(SelectColumns(keys=processed_column_names))
            )
            
        # Write the processed data to a bigquery table            
        write_processed = (
            processed | 'Write Processed' >> beam.io.WriteToBigQuery(
                table=known_args.processed_bq_table,
                dataset=known_args.bq_dataset,
                project=known_args.gcp_project,
                schema=processed_schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
        
        # Create a row for the header - a comma separated string
        header = ','.join(processed_column_names)
        
        csv_lines = (processed
                     | beam.Map(to_csv_row)
                    )
        
        write_to_csv = (
            csv_lines 
            | 'Write to CSV' >> beam.io.WriteToText(
                known_args.output_csv, 
                file_name_suffix='.csv', 
                header=header)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
