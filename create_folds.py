import pandas as pd
from sklearn import model_selection
import os

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold_, (train_idx, test_idx) in enumerate(kf.split(X= df, y=y)):
        print(fold_, train_idx.shape, test_idx.shape)
        df.loc[test_idx,"kfold"] = fold_
    df.to_csv("train_folds.csv", index=False)