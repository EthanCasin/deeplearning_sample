import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, classification_report
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def cleaner(df):
    """Cleans the dataset."""
    # map ordinal values
    mapper = [{'Has relevent experience': 1, 'No relevent experience': 0},   
          {'Phd': 5, 'Masters': 4, 'Graduate': 3, 'High School': 2,
           'Primary School': 1, np.nan: 0}, 
          {'>20': 22, '20': 21, '19': 20, '18': 19, '17': 18, '16': 17,
           '15': 16, '14': 15, '13': 14, '12': 13, '11': 12, '10': 11,
           '9': 10, '8': 9, '7': 8, '6': 7, '5': 6, '4': 5, '3': 4,
           '2': 3, '1': 2, '<1': 1, np.nan: 0},
          {np.nan : 0, '<10' : 1, '10/49' : 2, '50-99' : 3, '100-500' : 4,
          '500-999' : 5, '1000-4999' : 6, '5000-9999' : 7, '10000+' : 8},
          {'never' : 6, '>4' : 5, '4' : 4, '3' : 3, '2' : 2, '1' : 1,
           np.nan : 0}
          ]
    ord_cols = ['relevent_experience', 'education_level', 'experience',
            'company_size', 'last_new_job']

    for mapping, col in zip(mapper, ord_cols):
        df[col] = df[col].map(mapping)
    
    # fill in mode
    mode_cols = ['company_type', 'major_discipline',
                 'enrolled_university']
    for col in mode_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    # fill in not stated
    unknown = ['experience', 'gender']
    for col in unknown:
        df[col].fillna('Not Stated', inplace=True)
    return df


def class_perform(truth, predictions):
    """Returns a short report containing the accuracy, confusion matrix, 
    and overall classification metrics.
    """

    acc = accuracy_score(truth, predictions)  # get accuracy
    conf = confusion_matrix(truth, predictions)  # get confusion matrix
    rep = classification_report(truth, predictions)  # get report
    # draw report
    border = '=' * 55 + '\n'
    top = f'Accuracy: {acc*100:.2f}%' + '\n'
    div = '-' * 55 + '\n'
    mat = f'Confusion Matrix:\n{conf}\n'
    res = f'Classification Report:\n{rep}\n'
    return border + top + div + mat + div + rep + border


def kfold_cv(X, y, k_neighbors=5, model=None, n_splits=3,
             activation_function='relu'): 
    """Performs oversampling and K-Fold cross-validation. Oversampling is done
    for each fold to avoid bleeding out information. Improper oversampling can
    result to higher percieved accuracies. 
    
    To validate results, the AUC-ROC and the Recall must be shown. Moreover,
    the balanced accuracy must always be close to the AUC-ROC value. 
    
    Parameters:
    -----------
    X : array-like
        Input data. Preferably the entire dataset
    y : array-like
        Targets. 
    k_neighbors : int
        Number of k neighbors in conducting SMOTE. Default is 5.
    model : object-like, estimator
        Machine learning model to be used. Default is a Logistic Regression
        with random_state
    n_splits : int
        Number of K-Fold splits for cross validation. The higher the number, 
        the longer the runtime. Recommended values are in the range [3, 5].
    activation_function : string
        Activation function for the ELM model
    
    Returns:
    --------
    acc_mean : float
        Average (unbalanced) accuracy of the model
    bal_acc_mean : float
        Average balanced accuracy of the model
    auc_mean : float
        Average AUC-ROC of the model
    recall_mean : float
        Average recall of the model
    """
    # convert inputs to array
    X = np.array(X)
    y = np.array(y)
    
    # initialize smote
    smote = SMOTE(sampling_strategy='minority', k_neighbors=k_neighbors)
    
    # create placeholders
    auc = []
    recall = []
    accuracy = []
    bal_accuracy = []
    oo_fold = np.zeros(len(X))

    # initialize cross validation
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # begin operation
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f'Fold: {fold + 1} -> Starting...')
        
        # retrieve train and validation sets
        train_data, val_data = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # oversample training data
        train_upsample, y_upsample = smote.fit_resample(train_data, y_tr)
        
        # fit oversampled data to models
        if model is None: # check for input models
            model = GaussianNB()
            model.fit(train_upsample, y_upsample)
            oo_fold[val_idx] = model.predict_proba(val_data)[:, 1]
        else: # if the model is an ELM
            model.fit(train_upsample, y_upsample, activation_function)
            oo_fold[val_idx] = model.predict(val_data)

        
        # compute metrics
        # convert outputs into binary; some are floats
        aucrec = roc_auc_score(y_val, np.where(oo_fold[val_idx] > 0.5, 1, 0))
        rec = recall_score(y_val, np.where(oo_fold[val_idx] > 0.5, 1, 0))
        acc = accuracy_score(y_val, np.where(oo_fold[val_idx] > 0.5, 1, 0))
        bal_acc = balanced_accuracy_score(
            y_val, np.where(oo_fold[val_idx] > 0.5, 1, 0))
        
        # print results for each fold
        print(f'Accuracy: {acc}')
        print(f'Balanced Accuracy: {bal_acc}')
        print(f'Validation AUC-ROC: {aucrec}')
        print(f'Validation Recall: {rec}')
        
        # append results from each fold
        accuracy.append(acc)
        bal_accuracy.append(bal_acc)
        auc.append(aucrec)
        recall.append(rec)

    # get the average values of the metrics
    auc_mean = np.mean(auc)
    acc_mean = np.mean(accuracy)
    recall_mean = np.mean(recall)
    bal_acc_mean = np.mean(bal_accuracy)
    
    # print the average results
    print('=' * 40)
    print(f'Model Average Accuracy: {acc_mean}')
    print(f'Model Average Balanced Accuracy: {bal_acc_mean}')
    print(f'Model Average AUC-ROC: {auc_mean}')
    print(f'Model Average Recall: {recall_mean}')
    return acc_mean, bal_acc_mean, auc_mean, recall_mean


def get_callbacks():
    return [tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=70, restore_best_weights=True)]