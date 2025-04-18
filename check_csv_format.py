import csv
import sys

def is_valid_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            row_length = None
            for row in reader:
                if row_length is None:
                    row_length = len(row)
                elif len(row) != row_length:
                    print(f"Errore: Righe con numero di colonne diverso trovate.")
                    return False
            print("Il file è in formato CSV valido.")
            return True
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Utilizzo: python check_csv_format.py datasets/wikipedia_text_stats_grouped_silver_links.csv")
        is_valid_csv(file_path)
    else:
        file_path = sys.argv[1]
        is_valid_csv(file_path)
