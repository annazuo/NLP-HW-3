# README

## Part 1
For Validation Set:
    {'uas': 0.5748405017131534, 'las': 0.3480172399118694}

For Test Set:
    {'uas': 0.5742994241324775, 'las': 0.3490766501447936}

## Part 2
### Part 2.1
Note: run preprocessing and save files by running the following command line:
    python finetune_bert.py create_pp
See file included for en_gum_10.tsv

### Part 2.2
Note: run finetuning for the 3 lambdas and save the model states by running the following command line:
    python finetune_bert.py lambda_tuning
See files included for bert-parser-0.25.pt, bert-parser-0.5.pt, bert-parser-0.75.pt

For Lambda=0.25
    Final Validation Relative Position Accuracy: 0.7051347919840629
    Final Validation Dependency Accuracy: 0.8643302414001579

For Lambda=0.5
    Final Validation Relative Position Accuracy: 0.7412188010194627
    Final Validation Dependency Accuracy: 0.8563394005030497

For Lambda=0.75
    Final Validation Relative Position Accuracy: 0.751309021218361
    Final Validation Dependency Accuracy: 0.8222861255191293

## Part 3
For Validation Set:
    For Lambda=0.25


    For Lambda=0.5
        

    For Lambda=0.75


For Test Set:
