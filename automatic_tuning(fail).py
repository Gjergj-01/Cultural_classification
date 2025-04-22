import math
import numpy as np
import math
from datasets import load_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re
import json
import os
import time
import random
import ast

silver_dataset = pd.read_csv('datasets/silver_unicum.csv')
gold_dataset = pd.read_csv('datasets/gold_unicum.csv')

def to_quantili(train_dataset, val_dataset, in_column_name, out_column_name, n_quantili):
    #Taken a numerical column as input, <n_quantili> bins are created. 
    #The "quantilization" is performed based just on train_dataset values, to avoid data leakage
    #The bins are then applied both to the train and val dataset
    #Finally the old numerical columns are dropped
    
    bins_edges = pd.qcut(train_dataset[in_column_name], q=n_quantili, retbins=True)[1]
    train_dataset[out_column_name] = pd.cut(train_dataset[in_column_name], bins=bins_edges, include_lowest=True, duplicates='drop')
    val_dataset[out_column_name] = pd.cut(val_dataset[in_column_name], bins=bins_edges, include_lowest=True, duplicates='drop')

    train_dataset.drop(columns=in_column_name, inplace=True)
    val_dataset.drop(columns=in_column_name, inplace=True)

    return train_dataset, val_dataset

def feature_selection_with_rf(X_train, y_train, threshold=0.08):
    """
    Esegui la feature selection usando un modello Random Forest.
    Le caratteristiche con un'importanza inferiore al valore di soglia (threshold) vengono rimosse.

    Parameters:
    - X_train: DataFrame, features del training set
    - y_train: Series, target del training set
    - threshold: valore di soglia per determinare quali caratteristiche rimuovere

    Returns:
    - X_train_selected: DataFrame, X_train con solo le caratteristiche selezionate
    - selected_columns: lista delle colonne selezionate
    """
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_model.fit(X_train, y_train)
    
    feature_importances = rf_model.feature_importances_
    selected_columns = X_train.columns[feature_importances > threshold].tolist()
    
    X_train_selected = X_train[selected_columns]
    
    return X_train_selected, selected_columns

#drop unnecessary columns
unnecessary_columns = ['item', 'name', 'text_distribution', 'ref_distribution', 'sum_over_texts', 'avg_text', 'std_text']
silver_dataset = silver_dataset.drop(columns=unnecessary_columns)
gold_dataset = gold_dataset.drop(columns=unnecessary_columns)



def custom_quantili_search_wandb(silver_dataset_raw, gold_dataset_raw, label_columns, columns_to_quantili, n_iter=10, model_params_grid=None, project_name="quantili_tuning"):
    
    best_score = 0
    best_model = None
    best_config = None
    results = []

    for i in range(n_iter):
        # 🔹 Init un run W&B per ogni iterazione

        silver_dataset = silver_dataset_raw.copy()
        gold_dataset = gold_dataset_raw.copy()

        quantili_config = {col: random.choice([1, 2, 3, 4, 5, 6]) for col in columns_to_quantili}
        for col, n_q in quantili_config.items():
            silver_dataset, gold_dataset = to_quantili(silver_dataset, gold_dataset, col, "bin_" + col, n_q)

        categorical_columns = silver_dataset.columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(silver_dataset[categorical_columns])
        encoded_train = encoder.transform(silver_dataset[categorical_columns])
        encoded_eval = encoder.transform(gold_dataset[categorical_columns])
        encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns), index=silver_dataset.index)
        encoded_eval_df = pd.DataFrame(encoded_eval, columns=encoder.get_feature_names_out(categorical_columns), index=gold_dataset.index)

        dataset = silver_dataset.drop(columns=categorical_columns)
        dataset = pd.concat([dataset, encoded_train_df], axis=1)
        evaluation_dataset = gold_dataset.drop(columns=categorical_columns)
        evaluation_dataset = pd.concat([evaluation_dataset, encoded_eval_df], axis=1)

        X_train = dataset.drop(columns=label_columns)
        y_train = dataset[label_columns]
        X_test = evaluation_dataset.drop(columns=label_columns)
        y_test = evaluation_dataset[label_columns]

        # Esegui la feature selection per X_train
        X_train_selected, selected_columns = feature_selection_with_rf(X_train, y_train, threshold=0.01)

        # Aggiorna X_train con le colonne selezionate
        X_test_selected = X_test[selected_columns]

        if model_params_grid is None:
            model_params_grid = {
                'n_estimators': [50, 100, 150, 500, 1000],
            }

        rf_params = {k: random.choice(v) for k, v in model_params_grid.items()}
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        rf_model.fit(X_train_selected, y_train)

        y_pred = rf_model.predict(X_test_selected)
        score = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        print("ACCURACY: ", acc, "\t", "ITER"+str(i))

        results.append({
            "quantili_config": quantili_config,
            "model_params": rf_params,
            "f1_score": score,
            "accuracy": acc,
            "selected_columns": selected_columns
        })

        if score > best_score:
            best_score = score
            best_model = rf_model
            best_config = {
                "quantili_config": quantili_config,
                "model_params": rf_params,
                "selected_columns": selected_columns  # Aggiungiamo le colonne selezionate
            }

    print("\n✅ MIGLIORE CONFIGURAZIONE:")
    print(best_config)
    print(f"F1 Score migliore: {best_score:.4f}")

    return best_model, best_config, pd.DataFrame(results)

# Esegui la ricerca con feature selection
labels = ['label_cultural agnostic', 'label_cultural exclusive', 'label_cultural representative']
columns_to_quantili = ['std_text',
       'avg_text', 'len', 'entropy_text', 'gini_text', 'sum_over_texts',
        'std_ref', 'avg_ref', 'sum_over_ref', 'entropy_ref',
       'gini_ref']

model, best_config, df_results = custom_quantili_search_wandb(
    silver_dataset, gold_dataset,
    label_columns=labels,
    columns_to_quantili=columns_to_quantili,
    n_iter=50,
    project_name="my_experiment_quantili"
)
