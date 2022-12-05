from src.parsers.parser import Parser 
from src.dependency_parse import DependencyParse
from src.bert_parser_model import BertParserModel, subword_cleaning
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn.functional as F
import pickle
import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = True):
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

        # clean the output with the cleaning function made for hw part 2
        cleaned_output = subword_cleaning(output, tokenizer, sentence, tokens)
        cleaned_output_rel = torch.squeeze(cleaned_output[0])
        cleaned_output_deprel = torch.squeeze(cleaned_output[1])

        # if there's only one word, resize
        if len(cleaned_output_rel.size()) == 1:
            cleaned_output_rel = cleaned_output_rel.view(1, -1)
        if len(cleaned_output_deprel.size()) == 1:
            cleaned_output_deprel = cleaned_output_deprel.view(1, -1)

        # apply MST decoding
        if self.mst:
            # turn the logits from the matrix fo sentence_length x rel_vocab_size to log probabilities
            rel_log_probs = F.log_softmax(cleaned_output_rel, dim=1)

            # invert the rel_pos vocab mapper
            rel_pos_vocab_inv = {v: k for k, v in rel_pos_vocab.items()}

            # create a directed graph in which edges represent the potential head (token in sentence) nodes (token in sentence)
            G = nx.DiGraph()
            # loop through each token in the sentence representing the node (j) and put in the log prob for the potential head position (i)
            for j in range(len(tokens)):
                    G.add_node(j)
                    # (don't include self-connected nodes)
                    for i in range(len(tokens)):
                        if j != i:
                            # calculate the position of this potential head i
                            rel_pos_i = i - j
                            # find the index in the rel_log_probs matrix for this rel_pos_i
                            if rel_pos_i in rel_pos_vocab_inv.keys():
                                rel_pos_idx = rel_pos_vocab_inv[rel_pos_i]
                            # if this relative position doesn't exist, use the unk token weight
                            else:
                                rel_pos_idx = rel_pos_vocab_inv['unk']
                            # grab the weight at this position in the rel_log_probs matrix
                            weight = rel_log_probs[j][rel_pos_idx].item()
                            # add an edge to the graph with this weight
                            G.add_edge(i, j, weight=weight)
            # pass the contructed graph through the maximum spanning arborescence function
            MST = maximum_spanning_arborescence(G)

            # initialize a list for the head_preds with zeros (so if there's no valid edges, the node is a head with head=0)
            head_preds = ['0'] * len(tokens)

            # for the MST, loop through the edges to fill in the head predictions
            mst_edges = MST.edges
            for idx, edge in enumerate(mst_edges):
                head = edge[0]
                node = edge[1]
                head_preds[node] = str(head + 1)

            # for deprel, continue to use argmax
            deprel_preds_idx = torch.argmax(cleaned_output_deprel, dim=1)

            # convert the indices back into the true values
            deprel_preds = []
            for idx in range(len(deprel_preds_idx)):
                deprel_preds.append(deprel_vocab[deprel_preds_idx[idx].item()])

            # make a dictionary for the parse predictions
            dep_parse_dict = {'text': sentence, 'tokens': tokens, 'head': head_preds, 'deprel': deprel_preds}

            # return the parse in the DependencyParse format
            return DependencyParse.from_huggingface_dict(data_dict=dep_parse_dict)

        # apply argmax decoding
        else:
             # get the indices of the best prediction for each sentence (rel_pos predictions)
            rel_preds_idx = torch.argmax(cleaned_output_rel, dim=1)
            deprel_preds_idx = torch.argmax(cleaned_output_deprel, dim=1)

            # convert the indices back into the true values
            rel_preds = []
            deprel_preds = []
            for idx in range(len(rel_preds_idx)):
                rel_preds.append(rel_pos_vocab[rel_preds_idx[idx].item()])
                deprel_preds.append(deprel_vocab[deprel_preds_idx[idx].item()])

            # convert the relative position predictions to heads
            head_preds = []
            for idx, rel_pred in enumerate(rel_preds):
                if (rel_pred == 'pad') or (rel_pred == 'unk'):
                    head_preds.append('-1')
                elif rel_pred == 0:
                    head_preds.append('0')
                else:
                    head = idx + rel_pred + 1
                    head_preds.append(str(head))

            # make a dictionary for the parse predictions
            dep_parse_dict = {'text': sentence, 'tokens': tokens, 'head': head_preds, 'deprel': deprel_preds}

            # return the parse in the DependencyParse format
            return DependencyParse.from_huggingface_dict(data_dict=dep_parse_dict)