import torch
import torch.nn as nn

class BertParserModel(nn.Module):
    '''
    Pytorch class for finetuning Bert (copied over and renamed from finetune_bert.py)
    '''
    def __init__(self, d_hidden, n_classes_rel, n_classes_deprel, emb_model):
        super().__init__()
        self.rel_linear = nn.Linear(d_hidden, n_classes_rel)
        self.deprel_linear = nn.Linear(d_hidden, n_classes_deprel)
        self.emb_model = emb_model

    def forward(self, x):
        x = self.emb_model(**x).last_hidden_state
        x_rel = self.rel_linear(x)
        x_deprel = self.deprel_linear(x)
        return x_rel, x_deprel

def subword_cleaning(model_output, tokenizer, text, labels):
    '''
    Clean the model Bert tokenized outputs to remove subwords and hyphens (do on CPU to save GPU memory)
    '''
    # save the max length of the targets
    target_max_length = len(labels)

    # initialize lists of tensors for predicted rel_pos and deprel in batch
    rel_pos_preds = []
    deprel_preds = []

    # loop through each sentence within the batch (put in 1 here since for part 3, we are running this unbatched)
    for i in range(1):
        # grab the bert tokens and the tensor outputs (add CLS and SEP to match the tokenizer output)
        bert_tokens = ['CLS'] + tokenizer.tokenize(text) + ['SEP']
        rel_pos_pred = model_output[0][i]
        deprel_pred = model_output[1][i]

        # loop through each token and remove the subwords and hyphens from predictions
        new_bert_tokens = bert_tokens.copy()
        for j, token in enumerate(bert_tokens):
            # first, find the CLS and SEP tokens to remove
            if token == 'CLS':
                new_bert_tokens[j] = '__dummy__'
            if token == 'SEP':
                new_bert_tokens[j] = '__dummy__'
            # find the subwords and hyphen tokens to remove within the bert_tokens
            if '##' in token:
                new_bert_tokens[j] = '__dummy__'
            if token == '-':
                new_bert_tokens[j] = '__dummy__'
                if j != (len(bert_tokens)-1):
                    new_bert_tokens[j+1] = '__dummy__'

        # remove the tensors at the dummy indices within the predictions
        rel_pos_pred_list = []
        deprel_pred_list = []
        for j, token in enumerate(new_bert_tokens):
            if token != '__dummy__':
                rel_pos_pred_list.append(rel_pos_pred[j])
                deprel_pred_list.append(deprel_pred[j])
        rel_pos_pred = torch.stack(rel_pos_pred_list)
        deprel_pred = torch.stack(deprel_pred_list)

        # if the cleaned tensor is shorter than target_max_length, pad the tensors with zeros up so that the dimension is target_max_length x vocab size
        # (Note: this shouldn't be a big issue since majority of these zeros will be ignored in the loss function since they fall at pad indices)
        if target_max_length > rel_pos_pred.size(0):
            rel_pos_pred = torch.cat([rel_pos_pred.to('cpu'), torch.zeros((target_max_length - rel_pos_pred.size(0), rel_pos_pred.size(1))).to('cpu')])
            deprel_pred = torch.cat([deprel_pred.to('cpu'), torch.zeros((target_max_length - deprel_pred.size(0), deprel_pred.size(1))).to('cpu')])

        # if the cleaned tensor is longer than target_max_length, chop off the extra tensors so that the dimension matches target_max_length x vocab size
        # (Note: shouldn't effect loss that much since this just just an uncommon edge case)
        elif target_max_length < rel_pos_pred.size(0):
            rel_pos_pred = rel_pos_pred[:target_max_length, :]
            deprel_pred = deprel_pred[:target_max_length, :]

        rel_pos_preds.append(rel_pos_pred.to('cpu'))
        deprel_preds.append(deprel_pred.to('cpu'))

    # stack the tensors from each sentence of the batch back together
    rel_pos_preds_new = torch.stack(rel_pos_preds, dim=0)
    deprel_preds_new = torch.stack(deprel_preds, dim=0)

    return (rel_pos_preds_new, deprel_preds_new)