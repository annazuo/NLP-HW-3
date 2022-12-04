from src.parsers.parser import Parser 
from src.dependency_parse import DependencyParse
from src.bert_parser_model import BertParserModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import pickle

class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """Load your saved finetuned model using torch.load().

        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        self.checkpoint = torch.load(model_path)

    def parse(self, sentence: str, tokens: list):
        """Build a DependencyParse from the output of your loaded finetuned model.

        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # apply MST decoding
        if self.mst:
            pass
        # apply argmax decoding
        else:
            # reinitialize the model and the model parameters
            emb_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            d_hidden = emb_model.embeddings.word_embeddings.embedding_dim

            with open('rel_pos_vocab.pickle', 'rb') as f:
                rel_pos_vocab = pickle.load(f)
            with open('deprel_vocab.pickle', 'rb') as f:
                deprel_vocab = pickle.load(f)
            n_classes_rel = len(rel_pos_vocab)
            n_classes_deprel = len(deprel_vocab)

            model = BertParserModel(d_hidden, n_classes_rel, n_classes_deprel, emb_model)

            # load the model state and pass into the model
            model.load_state_dict(self.checkpoint['model_state_dict'])

            # initialize the pretrained tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

            # tokenize the input using DistilBERT
            tokenized_text = tokenizer(sentence, return_tensors='pt', padding=True)

            # pass through the model
            output = model(tokenized_text)

            print(len(tokenizer.tokenize(sentence)))
            print(tokenized_text['input_ids'].size())
