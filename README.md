# README

## Part 1
#### Short Description of Code:
To run the code, use the following command for the validation set:
    python eval_parser.py spacy
Also, use the following command for the test set:
    python eval_parser.py spacy --test

#### Results:
For Validation Set:
    {'uas': 0.5748405017131534, 'las': 0.3480172399118694}

For Test Set:
    {'uas': 0.5742994241324775, 'las': 0.3490766501447936}

## Part 2
### Part 2.1
#### Short Description of Code:
In the finetune_bert.py file, the functions can be run in command line by running in the following format:
    python finetune_bert.py function_name
In my code, the preprocessing function (which calls the get_parses function) takes in a data subset and preprocesses it. To run preprocessing on all three subsets, we run the create_pp function.

Run preprocessing save the files (train set, val set, test set, rel_pos vocab, deprel vocab) by running the following command line:
    python finetune_bert.py create_pp

#### Results:
See file included for en_gum_10.tsv

### Part 2.2
#### Short Description of Code:
In the finetune_bert.py file, the functions can be run in command line by running in the following format:
    python finetune_bert.py function_name
In my code, the run_model function maps the targets to indices, creates dataloaders, and trains and evaluates the model by calling the FinetuneBert model class, and the following functions: map_to_idx, pad_list_of_tensors, pad_collate_fn, and subword_cleaning. To run the model on all three lambdas, we run the lambda_tuning function.

Run finetuning for the 3 lambdas and save the model states by running the following command line:
    python finetune_bert.py lambda_tuning

#### Results:
See files included for bert-parser-0.25.pt, bert-parser-0.5.pt, bert-parser-0.75.pt

For Lambda=0.25
    Final Validation Relative Position Accuracy: 0.6738846541651689
    Final Validation Dependency Accuracy: 0.8483991252861763

For Lambda=0.5
    Final Validation Relative Position Accuracy: 0.7206644352497926
    Final Validation Dependency Accuracy: 0.8425029064018588

For Lambda=0.75
    Final Validation Relative Position Accuracy: 0.7359746412836106
    Final Validation Dependency Accuracy: 0.8120889659776388

## Part 3
#### Short Description of Code:
To run the code for Argmax Decoding, use the following command for the validation set:
    python eval_parser.py bert 
Also, use the following command for the test set:
    python eval_parser.py bert --test

To run the code for MST Decoding, use the following command for the validation set:
    python eal_parser.py bert --mst True
Also, use the following command for the test set:
    python eval_parser.py bert --test --mst True

#### Results:
For Argmax Decoding:
    For Validation Set:
        For Lambda=0.25:
            {'uas': 0.7151894713338056, 'las': 0.6801684958217957}

        For Lambda=0.5:
            {'uas': 0.7670725856515138, 'las': 0.7267341236821997}

        For Lambda=0.75:
            {'uas': 0.7859685655561269, 'las': 0.7231893096101568}

    For Test Set:
        Select the best lambda based on UAS.
        Best Lambda=0.75:
            {'uas': 0.7959862650311601, 'las': 0.7266512149163995}

For MST Decoding:
    For Validation Set:
        For Lambda=0.25:
            {'uas': 0.627935106407331, 'las': 0.5914486033180397}

        For Lambda=0.5:
            {'uas': 0.6737543844169256, 'las': 0.6331115535074764}

        For Lambda=0.75:
            {'uas': 0.6919389126132234, 'las': 0.6294586369525688}

    For Test Set:
        Select the best lambda based on UAS.
        Best Lambda=0.75:
            {'uas': 0.6918453636151792, 'las': 0.6223462936936909}