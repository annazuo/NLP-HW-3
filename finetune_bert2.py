"""
This file should implement all steps described in Part 2, and can be structured however you want.

Rather than using normal BERT, you should use distilbert-base-uncased. This will train faster.

We recommend training on a GPU, either by using HPC or running the command line commands on Colab.

Hints:
    * It will probably be helpful to save intermediate outputs (preprocessed data).
    * To save your finetuned models, you can use torch.save().
"""
import sys
from datasets import load_dataset
from src.dependency_parse import DependencyParse
import pandas as pd
import pickle
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import psutil
import humanize
import os
import GPUtil as GPU

def get_parses(subset='en_gum', dataset='train'):
    """
    Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py
    """
    # load in the dataset
    full_dataset = load_dataset("universal_dependencies", subset)

    # grab the dataset
    data = full_dataset[dataset]

    # make a list of dependency parse objects for each sentence in the subset
    dependency_parse_list = []
    for data_pt in data:
        dependency_parse = DependencyParse.from_huggingface_dict(data_dict=data_pt)
        dependency_parse_list.append(dependency_parse)

    return dependency_parse_list

def preprocessing(dataset='train', rel_pos_mapper=None, deprel_mapper=None):
    '''
    Preprocesses the dataset into rel_pos, deprel, etc. and creates vocabs and adds 'unk' tokens
    '''
    # get parses for the dataset
    dependency_parse_train = get_parses(dataset=dataset)
    
    data_df = pd.DataFrame(columns=['text', 'rel_pos', 'dep_label'])

    # make a relative position vocab and dependency vocab
    if dataset == 'train':
        rel_pos_vocab = ['pad', 'unk']
        deprel_vocab = ['pad', 'unk']

    all_tokens = []
    # loop through each example
    for sentence in dependency_parse_train:
        text = sentence.text
        tokens = sentence.tokens
        heads = sentence.heads
        deprel = sentence.deprel

        rel_positions = []
        # compute the relative positions
        for token_idx in range(len(tokens)):
            if deprel[token_idx] == 'root':
                rel_pos = 0
            else:
                rel_pos = (int(heads[token_idx]) - 1) - token_idx
            
            # if we're preprocessing the train set, add the rel_pos and deprel to the vocab
            if dataset == 'train':
                rel_pos_vocab.append(rel_pos)
                deprel_vocab.append(deprel[token_idx])
            # if we're preprocessing the val or test set, put unk if not in vocab
            else:
                if rel_pos not in rel_pos_mapper.values():
                    rel_pos = 'unk'
                if deprel[token_idx] not in deprel_mapper.values():
                    deprel[token_idx] = 'unk'
        
            rel_positions.append(rel_pos)

        all_tokens.append(tokens)

        # add a row to the dataframe for this sentence
        data_df.loc[len(data_df.index)] = [text, rel_positions, deprel] 

    if dataset == 'train':
        # write out to tsv
        with open('en_gum_10.tsv','w') as write_tsv:
            write_tsv.write(data_df[:10].to_csv(sep='\t', index=False, header=False))

        # only grab the unique items for the vocab
        rel_pos_vocab = list(sorted(set(rel_pos_vocab), key=rel_pos_vocab.index))
        deprel_vocab = list(sorted(set(deprel_vocab), key=deprel_vocab.index))

        # make mappers of the vocab items to indices to return
        rel_pos_mapper = {k: v for k, v in enumerate(rel_pos_vocab)}
        deprel_mapper = {k: v for k, v in enumerate(deprel_vocab)}
    else:
        rel_pos_mapper = None
        deprel_mapper = None

    # also, create a preprocessed dataset to return
    preprocessed_data = []
    for i in range(data_df.shape[0]):
        preprocessed_data.append({'text': data_df.text[i],
                                  'rel_pos': data_df.rel_pos[i],
                                  'deprel': data_df.dep_label[i],
                                  'tokens': all_tokens[i]})

    return preprocessed_data, rel_pos_mapper, deprel_mapper

def create_pp():
    '''
    Function to run preprocessing on all three data splits
    '''
    # create the preprocessed datasets and mappers
    preproc_train, train_rel_pos_mapper, train_deprel_mapper = preprocessing(dataset='train')
    preproc_val, _, _ = preprocessing(dataset='validation', rel_pos_mapper=train_rel_pos_mapper, deprel_mapper=train_deprel_mapper)
    preproc_test, _, _ = preprocessing(dataset='test', rel_pos_mapper=train_rel_pos_mapper, deprel_mapper=train_deprel_mapper)

    # pick the datasets and the mappers
    with open('train_pp.pickle', 'wb') as handle:
        pickle.dump(preproc_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('val_pp.pickle', 'wb') as handle:
        pickle.dump(preproc_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('test_pp.pickle', 'wb') as handle:
        pickle.dump(preproc_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('rel_pos_vocab.pickle', 'wb') as handle:
        pickle.dump(train_rel_pos_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('deprel_vocab.pickle', 'wb') as handle:
        pickle.dump(train_deprel_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)

class FinetuneBert(nn.Module):
    '''
    Pytorch class for finetuning Bert
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

def map_to_idx(dataset, rel_pos_mapper, deprel_mapper):
    '''
    Takes a data split and maps the rel_pos and deprel components to their indices within the training vocab
    '''
    # invert the mappings so that new it's value: index
    rel_pos_mapper_inv = {v: k for k, v in rel_pos_mapper.items()}
    deprel_mapper_inv = {v: k for k, v in deprel_mapper.items()}

    # loop through each item in the dataset and replace the rel_pos and deprel with tensors of indices
    for i, item in enumerate(dataset):
        rel_pos = item['rel_pos']
        deprel = item['deprel']

        rel_pos_new = []
        deprel_new = []
        for j in range(len(rel_pos)):
            rel_i = rel_pos_mapper_inv[rel_pos[j]]
            rel_pos_new.append(rel_i)
            deprel_i = deprel_mapper_inv[deprel[j]]
            deprel_new.append(deprel_i)
        # turnt the lists into tensors
        rel_pos_new = torch.Tensor(rel_pos_new).view(1, -1)
        deprel_new = torch.Tensor(deprel_new).view(1, -1)
        # replace the rel_pos and deprel with the new tensors
        dataset[i]['rel_pos'] = rel_pos_new
        dataset[i]['deprel'] = deprel_new
    return dataset

def pad_list_of_tensors(list_of_tensors, pad_id):
    '''
    Function to pad the rel_pos and deprel components of each batch to the max length within the batch
    '''
    # calculate the max length tensor in this batch
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    
    for t in list_of_tensors:
        padded_tensor = torch.cat([t, torch.tensor([[pad_id]*(max_length - t.size(-1))], dtype=torch.long).view(1, -1)], dim = -1)
        padded_list.append(padded_tensor)
        
    padded_tensor = torch.cat(padded_list, dim=0)
    
    return padded_tensor

def pad_collate_fn(batch):
    '''
    Collate function for the dataloader
    '''
    # get lists of the rel_pos and deprel for each sentence
    rel_pos = [s['rel_pos'] for s in batch]
    deprel = [s['deprel'] for s in batch]

    # read the vocabs back in
    with open('rel_pos_vocab.pickle', 'rb') as f:
        rel_pos_vocab = pickle.load(f)
    with open('deprel_vocab.pickle', 'rb') as f:
        deprel_vocab = pickle.load(f)

    # invert the mappings
    rel_pos_mapper_inv = {v: k for k, v in rel_pos_vocab.items()}
    deprel_mapper_inv = {v: k for k, v in deprel_vocab.items()}

    # find the pad indices
    pad_id_rel = rel_pos_mapper_inv['pad']
    pad_id_deprel = deprel_mapper_inv['pad']

    # pad the tensors to the max length in the batch
    rel_pos_tensor = pad_list_of_tensors(rel_pos, pad_id=pad_id_rel)
    deprel_tensor = pad_list_of_tensors(deprel, pad_id=pad_id_deprel)
    
    return [s['text'] for s in batch], rel_pos_tensor, deprel_tensor, [s['tokens'] for s in batch]

def subword_cleaning(model_output, tokenizer, text, rel_pos_labels, deprel_labels):
    '''
    Clean the model Bert tokenized outputs to remove subwords and hyphens (do on CPU to save GPU memory)
    '''
    # save the max length of the targets
    target_max_length = rel_pos_labels.size(1)

    # initialize lists of tensors for predicted rel_pos and deprel in batch
    rel_pos_preds = []
    deprel_preds = []

    # loop through each sentence within the batch
    for i in range(len(text)):
        # grab the bert tokens and the tensor outputs
        bert_tokens = tokenizer.tokenize(text[i])
        rel_pos_pred = model_output[0][i]
        deprel_pred = model_output[1][i]

        # loop through each token and remove the subwords and hyphens from predictions
        new_bert_tokens = bert_tokens.copy()
        for j, token in enumerate(bert_tokens):
            # find the tokens to remove within the bert_tokens
            if '##' in token:
                new_bert_tokens[j] = '__dummy__'
            if token == '-':
                new_bert_tokens[j] = '__dummy__'
                if j != (len(bert_tokens)-1):
                    new_bert_tokens[j+1] = '__dummy__'

        # remove the tensors at the dummy indices within the predictions
        rel_pos_pred_list = []
        deprel_pred_list = []
        # cleaned_bert_tokens = [] # DELETE LATER
        for j, token in enumerate(new_bert_tokens):
            if token != '__dummy__':
                rel_pos_pred_list.append(rel_pos_pred[j].view(1, -1))
                deprel_pred_list.append(deprel_pred[j].view(1, -1))
                # cleaned_bert_tokens.append(token) # DELETE LATER
        rel_pos_pred = torch.cat(rel_pos_pred_list)
        deprel_pred = torch.cat(deprel_pred_list)

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
            # cleaned_bert_tokens = cleaned_bert_tokens[:target_max_length] # DELETE LATER

        rel_pos_preds.append(rel_pos_pred.to('cpu'))
        deprel_preds.append(deprel_pred.to('cpu'))

    # stack the tensors from each sentence of the batch back together
    rel_pos_preds_new = torch.stack(rel_pos_preds, dim=0)
    deprel_preds_new = torch.stack(deprel_preds, dim=0)

    return (rel_pos_preds_new, deprel_preds_new)

def run_model(batch_size=32, num_epochs=3, lr=0.0001, lambda_loss=0.25, max_norm=1.0, save_model=False):
    '''
    Function to run the finetuning Bert model
    '''
    # clear GPU and print current usage
    torch.cuda.empty_cache()
    # GPUs = GPU.getGPUs()
    # gpu = GPUs[0]
    # def printm():
    #     process = psutil.Process(os.getpid())
    #     print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    #     print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    # printm()

    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

    # create distilBERT objects
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    emb_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    # read the datasets and vocabs back in
    with open('train_pp.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('val_pp.pickle', 'rb') as f:
        val = pickle.load(f)
    with open('test_pp.pickle', 'rb') as f:
        test = pickle.load(f)
    with open('rel_pos_vocab.pickle', 'rb') as f:
        rel_pos_vocab = pickle.load(f)
    with open('deprel_vocab.pickle', 'rb') as f:
        deprel_vocab = pickle.load(f)

    # invert the mappings
    rel_pos_mapper_inv = {v: k for k, v in rel_pos_vocab.items()}
    deprel_mapper_inv = {v: k for k, v in deprel_vocab.items()}

    print('train', train)

    # for the datasets, turn the rel_pos and deprel into their respective indices
    train = map_to_idx(train, rel_pos_vocab, deprel_vocab)
    val = map_to_idx(val, rel_pos_vocab, deprel_vocab)
    test = map_to_idx(test, rel_pos_vocab, deprel_vocab)

    # make dataloader objects
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # parameters for the model
    d_hidden = emb_model.embeddings.word_embeddings.embedding_dim

    # define the model, optimizer, and criterion
    model = FinetuneBert(d_hidden=d_hidden, n_classes_rel=len(rel_pos_vocab), n_classes_deprel=len(deprel_vocab), emb_model=emb_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=rel_pos_mapper_inv['pad'])

    train_losses = []
    train_accs_rel = []
    train_accs_deprel = []
    val_losses = []
    val_accs_rel = []
    val_accs_deprel = []
    # run a training loop for the model
    for epoch in range(num_epochs):
        # first, model training per epoch
        model.train()
        train_losses_batch = []
        train_acc_rel_batch = []
        train_acc_deprel_batch = []
        for idx, (text, rel_pos, deprel, tokens) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            # tokenize the input using DistilBERT
            tokenized_text = tokenizer(text, return_tensors='pt', padding=True).to(device)

            # pass the encoded input through the model
            output = model(tokenized_text)

            # take each sentence in the batch, and remove the subwords and hyphens created by bert tokenizer
            cleaned_output = subword_cleaning(output, tokenizer, text, rel_pos, deprel)

            # permute the outputs so that they are of size batch_size x vocab_size x max_length
            cleaned_output_rel = cleaned_output[0].permute(0, 2, 1)
            cleaned_output_deprel = cleaned_output[1].permute(0, 2, 1)

            loss_rel = criterion(cleaned_output_rel, rel_pos.type(torch.LongTensor))
            loss_deprel = criterion(cleaned_output_deprel, deprel.type(torch.LongTensor))

            # get the indices of the best prediction for each sentence in batch
            rel_preds = torch.argmax(cleaned_output_rel, dim=1)
            deprel_preds = torch.argmax(cleaned_output_deprel, dim=1)

            # compute the avg accuracy for this batch (don't count padding when computing accuracy)
            acc_rel_counts = 0
            acc_rel_total = 0
            for j in range(rel_pos.size(0)):
                for k in range(rel_pos.size(1)):
                    label_item = rel_pos[j][k]
                    pred_item = rel_preds[j][k]
                    if label_item != 0:
                        acc_rel_total += 1
                        if label_item == pred_item: 
                            acc_rel_counts += 1

            acc_deprel_counts = 0
            acc_deprel_total = 0
            for j in range(deprel.size(0)):
                for k in range(deprel.size(1)):
                    label_item = deprel[j][k]
                    pred_item = deprel_preds[j][k]
                    if label_item != 0:
                        acc_deprel_total += 1
                        if label_item == pred_item: 
                            acc_deprel_counts += 1

            train_acc_rel = acc_rel_counts/acc_rel_total
            train_acc_deprel = acc_deprel_counts/acc_deprel_total
            train_acc_rel_batch.append(train_acc_rel)
            train_acc_deprel_batch.append(train_acc_deprel)
            
            # compute total loss with lambda
            loss_tot = (lambda_loss*loss_rel) + ((1-lambda_loss)*loss_deprel)
            loss_tot.backward()

            # implement gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)

            optimizer.step()

            train_losses_batch.append(loss_tot)

        # next, evaluate the model 
        model.eval()
        val_losses_batch = []
        val_acc_rel_batch = []
        val_acc_deprel_batch = []
        for idx, (text, rel_pos, deprel, tokens) in tqdm(enumerate(val_dataloader)):
            # tokenize the input using DistilBERT
            tokenized_text = tokenizer(text, return_tensors='pt', padding=True).to(device)

            # pass the encoded input through the model
            output = model(tokenized_text)

            # take each sentence in the batch, and remove the subwords and hyphens created by bert tokenizer
            cleaned_output = subword_cleaning(output, tokenizer, text, rel_pos, deprel)

            # permute the outputs so that they are of size batch_size x vocab_size x max_length
            cleaned_output_rel = cleaned_output[0].permute(0, 2, 1)
            cleaned_output_deprel = cleaned_output[1].permute(0, 2, 1)

            loss_rel = criterion(cleaned_output_rel, rel_pos.type(torch.LongTensor))
            loss_deprel = criterion(cleaned_output_deprel, deprel.type(torch.LongTensor))

            # get the indices of the best prediction for each sentence in batch
            rel_preds = torch.argmax(cleaned_output_rel, dim=1)
            deprel_preds = torch.argmax(cleaned_output_deprel, dim=1)

            # compute the avg accuracy for this batch (don't count padding when computing accuracy)
            acc_rel_counts = 0
            acc_rel_total = 0
            for j in range(rel_pos.size(0)):
                for k in range(rel_pos.size(1)):
                    label_item = rel_pos[j][k]
                    pred_item = rel_preds[j][k]
                    if label_item != 0:
                        acc_rel_total += 1
                        if label_item == pred_item: 
                            acc_rel_counts += 1

            acc_deprel_counts = 0
            acc_deprel_total = 0
            for j in range(deprel.size(0)):
                for k in range(deprel.size(1)):
                    label_item = deprel[j][k]
                    pred_item = deprel_preds[j][k]
                    if label_item != 0:
                        acc_deprel_total += 1
                        if label_item == pred_item: 
                            acc_deprel_counts += 1

            val_acc_rel = acc_rel_counts/acc_rel_total
            val_acc_deprel = acc_deprel_counts/acc_deprel_total
            val_acc_rel_batch.append(val_acc_rel)
            val_acc_deprel_batch.append(val_acc_deprel)

            # compute total loss with lambda
            loss_tot = (lambda_loss*loss_rel) + ((1-lambda_loss)*loss_deprel)

            val_losses_batch.append(loss_tot)

        # append losses and accuracies
        avg_train_loss = sum(train_losses_batch)/len(train_losses_batch)
        train_losses.append(avg_train_loss)
        avg_train_acc_rel = sum(train_acc_rel_batch)/len(train_acc_rel_batch)
        train_accs_rel.append(avg_train_acc_rel)
        avg_train_acc_deprel = sum(train_acc_deprel_batch)/len(train_acc_deprel_batch)
        train_accs_deprel.append(avg_train_acc_deprel)

        avg_val_loss = sum(val_losses_batch)/len(val_losses_batch)
        val_losses.append(avg_val_loss)
        avg_val_acc_rel = sum(val_acc_rel_batch)/len(val_acc_rel_batch)
        val_accs_rel.append(avg_val_acc_rel)
        avg_val_acc_deprel = sum(val_acc_deprel_batch)/len(val_acc_deprel_batch)
        val_accs_deprel.append(avg_val_acc_deprel)

        # print losses and accuracies
        print('Training Loss for Epoch {}: {}'.format(epoch, avg_train_loss))
        print('Training Relative Position Accuracy for Epoch {}: {}'.format(epoch, avg_train_acc_rel))
        print('Training Dependency Accuracy for Epoch {}: {}'.format(epoch, avg_train_acc_deprel))
        print('')
        print('Validation Loss for Epoch {}: {}'.format(epoch, avg_val_loss))
        print('Validation Relative Position Accuracy for Epoch {}: {}'.format(epoch, avg_val_acc_rel))
        print('Validation Dependency Accuracy for Epoch {}: {}'.format(epoch, avg_val_acc_deprel))
        print('')
        # printm()
        # print('')

    # if the save model flag is on, save the model to the current directory
    if save_model:
        torch.save({'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'train_accs_rel': train_accs_rel,
                    'train_accs_deprel': train_accs_deprel,
                    'val_losses': val_losses,
                    'val_accs_rel': val_accs_rel,
                    'val_accs_deprel': val_accs_deprel}, 'bert-parser-{}.pt'.format(lambda_loss))
    return

def lambda_tuning(lambdas=[0.25, 0.5, 0.75]):
    for lambda_loss in lambdas:
        print('Training and Validation for Lambda={}'.format(lambda_loss))
        run_model(lambda_loss=lambda_loss, save_model=True)

if __name__ == '__main__':
    globals()[sys.argv[1]]()