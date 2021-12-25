import argparse
import os
import sys
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import roc_auc_score
import csv

print(f'pandas version {pd.__version__}')

print(f'Sklearn version {sklearn.__version__}')

try:
    from sklearn.externals import joblib
except:
    import joblib


def run(arguments):
    test_file = None
    train_file = None
    validation_file = None
    joblib_file = "LR_model.pkl"

    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')
    args = parser.parse_args(arguments)

    Train = False
    Test = False
    Validation = False

    if args.test != None:
        print(f"Test file with attributes to predict: {args.test}")
        Test = True
        
    else:
        if args.train != None:
            print(f"Training data file: {args.train}")
            Train = True

        if args.validation != None:
            print(f"Validation data file: {args.validation}")
            Validation = True

    if Train and Validation:
        file_train = pd.read_csv(args.train, quotechar='"', usecols=[0, 1, 2, 3],
                                 dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 6))
        corpora = file_train['text_'].astype(str).values.tolist()
        vectorizer.fit(corpora)
        X = vectorizer.transform(corpora)
        with open('vectorizer.pk', 'wb') as fout:
            pickle.dump(vectorizer, fout)
        file_validation = pd.read_csv(args.validation, quotechar='"', usecols=[0, 1, 2, 3],
                                      dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        corpora2 = file_validation['text_'].astype(str).values.tolist()
        XV = vectorizer.transform(corpora2)
        best_accuracy = [0, 0]
        print()
        for c in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
            lr = LogisticRegression(penalty="l2", tol=0.001, C=c, fit_intercept=True, solver='liblinear',
                                    intercept_scaling=1, max_iter=5000)
            lr.fit(X, file_train['real review?'])
            y_predicted = lr.predict_proba(XV.toarray())[:, 1]
            accuracy = roc_auc_score(file_validation['real review?'], y_predicted)
            print(f'C: {c}, AUC: {accuracy}')
            if accuracy > best_accuracy[0]:
                joblib.dump(lr, joblib_file)
                best_accuracy[0] = accuracy
                best_accuracy[1] = c
        print()
        print(f'Best AUC: {best_accuracy[0]}, C = {best_accuracy[1]}')

    elif Test:
        vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
        file_test = pd.read_csv(args.test, quotechar='"', usecols=[0, 1, 2, 3],
                                dtype={'ID': int, 'real review?': int, 'category': str, 'rating': int, 'text_': str})
        corpora = file_test['text_'].astype(str).values.tolist()
        X = vectorizer.transform(corpora)
        lr = joblib.load(joblib_file)
        y_hat = lr.predict_proba(X.toarray())[:, 1]
        with open('predictions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'real review?'])
            for i, y in enumerate(y_hat):
                writer.writerow([i, y])

    else:
        print("Training requires both training and validation data files. Test just requires test attributes.")


if __name__ == "__main__":
    run(sys.argv[1:])
