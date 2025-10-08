

def f1_score(y_true, y_pred):  # labels in {-1,1}
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==-1) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==-1))
    if tp==0: 
        return 0.0
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0