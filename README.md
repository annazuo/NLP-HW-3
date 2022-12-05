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
    Final Validation Relative Position Accuracy: 0.6738846541651689
    Final Validation Dependency Accuracy: 0.8483991252861763

For Lambda=0.5
    Final Validation Relative Position Accuracy: 0.7206644352497926
    Final Validation Dependency Accuracy: 0.8425029064018588

For Lambda=0.75
    Final Validation Relative Position Accuracy: 0.7359746412836106
    Final Validation Dependency Accuracy: 0.8120889659776388

## Part 3
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
            {'uas': 0.6270472248723779, 'las': 0.5908511837271458}

        For Lambda=0.5:
            {'uas': 0.6729592700405519, 'las': 0.6324549475374733}

        For Lambda=0.75:
            {'uas': 0.6917713665581569, 'las': 0.6290348717052857}

    For Test Set:
        Select the best lambda based on UAS.
        Best Lambda=0.75:
            {'uas': 0.6919016000785577, 'las': 0.6225292247864233}