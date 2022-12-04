from argparse import ArgumentParser
from collections import defaultdict

from src.dependency_parse import DependencyParse
from src.parsers.bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics
import numpy as np

from datasets import load_dataset


def get_parses(subset: str, test: bool = False):
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # load in the dataset
    full_dataset = load_dataset("universal_dependencies", subset)

    # choose the dataset to grab
    if test:
        dataset = 'test'
    else:
        dataset = 'validation'

    # grab the validation or test dataset
    data = full_dataset[dataset]

    # make a list of dependency parse objects for each sentence in the subset
    dependency_parse_list = []
    for data_pt in data:
        dependency_parse = DependencyParse.from_huggingface_dict(data_dict=data_pt)
        dependency_parse_list.append(dependency_parse)

    return dependency_parse_list


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("method", choices=["spacy", "bert"])
    arg_parser.add_argument("--data_subset", type=str, default="en_gum")
    arg_parser.add_argument("--test", action="store_false")
    
    # SpaCy parser arguments.
    arg_parser.add_argument("--model_name", type=str, default="en_core_web_sm")

    # BERT parser arguments
    arg_parser.add_argument("--model_path", type=str, default="bert-parser-0.25.pt")
    arg_parser.add_argument("--mst", type=bool, default=False)

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method == "spacy":
        parser = SpacyParser(args.model_name)
    elif args.method == "bert":
        parser = BertParser(model_path=args.model_path, mst=args.mst)
    else:
        raise ValueError("Unknown parser")

    cum_metrics = defaultdict(list)
    for gold in get_parses(args.data_subset, test=args.test):
        pred = parser.parse(gold.text, gold.tokens)
        for metric, value in get_metrics(pred, gold).items():
            cum_metrics[metric].append(value)
    
    print({metric: np.mean(data) for metric, data in cum_metrics.items()})
