from src.dependency_parse import DependencyParse


def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    # get the predicted and labeled heads and dependencies
    pred_heads = predicted.heads
    label_heads = labeled.heads
    pred_deprel = predicted.deprel
    label_deprel = labeled.deprel

    uas_counter = 0
    las_counter = 0
    # loop through each head/token in the dataset
    for i in range(len(pred_heads)):
        if pred_heads[i] == label_heads[i]:
            uas_counter += 1
        if (pred_heads[i] == label_heads[i]) and (pred_deprel[i] == label_deprel[i]):
            las_counter += 1
    
    # compute the uas for this sentence
    uas = uas_counter/len(pred_heads)

    # compute the las for this sentence
    las = las_counter/len(pred_heads)

    return {
        "uas": uas,
        "las": las}
