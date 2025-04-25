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
from bs4 import BeautifulSoup

def extract_refs(lang, wikipedia_url):
    title = wikipedia_url.split('/')[-1]
    lang = lang.replace('_', '-')
    base_api_url = f"https://{lang}.wikipedia.org/w/api.php"
    
    try:

        parse_params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text",
            "redirects": 1
        }

        res_html = requests.get(base_api_url, params=parse_params, timeout=5).json()
        html = res_html.get("parse", {}).get("text", {}).get("*", "")

        soup = BeautifulSoup(html, 'html.parser')
        ref_container = soup.find(class_="references")
        if ref_container:
            # Assumiamo che sia una lista "li"
            return len(ref_container.find_all("li", recursive=False))
        else:
            return 0
        return 0

    except requests.exceptions.RequestException as e:
        print("Errore per link", wikipedia_url)
        return 0
    
def extract_entity_id(url):
    return url.strip().split("/")[-1]

def process_entity(row, processed_keywords):
    item = row['entity']
    dict_url = ast.literal_eval(row['lang_url_map'])

    if item in processed_keywords:
        return None

    results = [item, ""]
    ref_distribution = []

    for key in dict_url:
        ref_num = extract_refs(key, dict_url[key])
        if ref_num == 0:
            continue
        ref_distribution.append(ref_num)

    if not ref_distribution:
        return None

    l = len(ref_distribution)
    mean = sum(ref_distribution) / l
    std = math.sqrt(sum((x - mean) ** 2 for x in ref_distribution) / l)

    results.append(ref_distribution)
    results.append(std)
    results.append(mean)

    return results

def save_wikipedia_text_parallel(file_name):
    output_file_name = "wikipedia_references_stats_" + file_name
    # Carica keyword già trattate
    if os.path.exists(output_file_name):
        existing_df = pd.read_csv(output_file_name)
        processed_keywords = existing_df['entity'].unique().tolist()
        print(len(processed_keywords), "keywords già trattate.")
    else:
        processed_keywords = []
        # Scriviamo l'header se il file non esiste
        with open(output_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['entity', 'ref_distribution', 'ref_std', 'ref_avg'])

    data = pd.read_csv("datasets/" + file_name)
    df = pd.DataFrame(data, columns=['entity', 'lang_url_map'])

    # Parallelizza il processo
    with ProcessPoolExecutor() as executor:
        future_to_entity = {
            executor.submit(process_entity, row, processed_keywords): row['entity']
            for _, row in df.iterrows()
        }

        #Barra di donwload
        for future in tqdm(as_completed(future_to_entity), total=len(future_to_entity)):
            result = future.result()
            if result:
                # Scrivi subito sul file ogni risultato ricevuto
                with open(output_file_name, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(result)

    print("Processo completato")


#entity, engtext, distribution, std, avg
if __name__ == '__main__':
    save_wikipedia_text_parallel('grouped_silver_links.csv')