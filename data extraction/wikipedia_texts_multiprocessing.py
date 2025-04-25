import math
import numpy as np
import math
from datasets import load_dataset
import pandas as pd
import re
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import requests
from wikidata.client import Client
from itertools import islice
import ast
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_text(lang, wikipedia_url):
    
    title = wikipedia_url.split('/')[-1]
    #print("english wikipedia title:", title)
    lang = lang.replace('_', '-')

    # Use Wikipedia's API to get plain text of the article
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1
    }

    try:
        res = requests.get(api_url, params=params, timeout=5).json()
        page = next(iter(res["query"]["pages"].values())) #see below the dictionary, we create an iterable and pick the first and only item
        text = page.get("extract", "") # we get the "extract field"
        if lang == 'en':
            return (text, len(text))
        return ("", len(text))
    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta: {e} per link {wikipedia_url}")
        return ("", 0)
    
def extract_entity_id(url):
    return url.strip().split("/")[-1]

def process_entity(row, processed_keywords):
    item = row['entity']
    dict_url = ast.literal_eval(row['lang_url_map'])

    if item in processed_keywords:
        return None

    results = [item, ""]
    lang_distribution = []

    for key in dict_url:
        text, length = extract_text(key, dict_url[key])
        if length == 0:
            continue
        if key == 'en':
            clean_text = text.replace('\n', ' ').replace('\t', ' ').strip()
            results[1] = clean_text
        lang_distribution.append(length)

    if not lang_distribution:
        return None

    l = len(lang_distribution)
    mean = sum(lang_distribution) / l
    std = math.sqrt(sum((x - mean) ** 2 for x in lang_distribution) / l)

    results.append(lang_distribution)
    results.append(std)
    results.append(mean)

    return results

def save_wikipedia_text_parallel(file_name):
    output_file_name = "wikipedia_text_stats_" + file_name

    # Carica keyword già trattate
    if os.path.exists(output_file_name):
        existing_df = pd.read_csv(output_file_name)
        processed_keywords = existing_df['entity'].unique().tolist()
        print(len(processed_keywords), "keywords già trattate.")
    else:
        processed_keywords = []

    data = pd.read_csv("datasets/" + file_name)
    df = pd.DataFrame(data, columns=['entity', 'lang_url_map'])

    # Parallelizza il processo
    with ProcessPoolExecutor() as executor:
        future_to_entity = {
            executor.submit(process_entity, row, processed_keywords): row['entity']
            for _, row in df.iterrows()
        }

        for future in tqdm(as_completed(future_to_entity), total=len(future_to_entity)):
            result = future.result()
            if result:
                # Scrivi subito sul file ogni risultato ricevuto
                with open(output_file_name, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(result)

    print("Processing completato")


#entity, engtext, distribution, std, avg
if __name__ == '__main__':
    save_wikipedia_text_parallel('grouped_silver_links.csv')