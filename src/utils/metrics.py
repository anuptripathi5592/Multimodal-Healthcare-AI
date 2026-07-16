from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(predictions, targets):
    acc = accuracy_score(targets, predictions)
    prec = precision_score(targets, predictions, average='weighted')
    rec = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
