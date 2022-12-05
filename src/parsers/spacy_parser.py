from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse 
import spacy

class SpacyParser(Parser):

    def __init__(self, model_name: str):
        # load in the model
        self.model = spacy.load(model_name)


    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Use the specified spaCy model to parse the sentence.py.

        The function should return the parse in the Dependency format.

        You should refer to the spaCy documentation for how to use the model for dependency parsing.
        """
        # initialize a dictionary for the parse predictions
        dep_parse_dict = {'text': sentence, 'tokens': [], 'head': [], 'deprel':[]}

        # make a spacy doc
        doc = spacy.tokens.doc.Doc(self.model.vocab, words=tokens)

        # parse the doc
        parse = self.model(doc)

        # get the tokens, head indices, and dependencies for each token
        for token in parse:
            # if the token is the root of the sentence, set head ID to 0
            if token.head.i == token.i:
                head_idx = str(0)
            # otherwise, corrected the index mismatch between spacey and UD
            else:
                head_idx =  str(token.head.i + 1)

            dep_parse_dict['tokens'].append(token.text)
            dep_parse_dict['head'].append(head_idx)
            dep_parse_dict['deprel'].append(token.dep_)

        # return the parse in the DependencyParse format
        return DependencyParse.from_huggingface_dict(data_dict=dep_parse_dict)
