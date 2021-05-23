import argparse
import sys
sys.path.append('../')
import numpy as np
import os
import json
from glob import glob

from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge, add_result


def main(args):
    supert = Supert()

    output = {}
    for instance_dir in glob(f'{args.input_dir}/*'):
        if not os.path.isdir(instance_dir):
            continue

        instance_id = os.path.basename(instance_dir)

        reader = CorpusReader(instance_dir)
        docs = reader()
        filenames, summaries = reader.readSummaries()

        supert.load_documents(docs)
        scores = supert(summaries)

        output[instance_id] = {filename: score for filename, score in zip(filenames, scores)}

    with open(args.output_jsonl, 'w') as out:
        out.write(json.dumps(output))



if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_dir')
    argp.add_argument('output_jsonl')
    args = argp.parse_args()
    main(args)
