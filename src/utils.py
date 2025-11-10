# src/utils.py

import random
import os
import numpy as np
import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict

# matplotlib és seaborn importálása az ábrákhoz
# A 'Agg' backend-et használjuk, ami nem igényel grafikus felületet
# (hasznos szerveren futtatáskor)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns

# --- 1. Reprodukálhatóság ---

def set_seed(seed: int = 42):
    """
    Globális seed beállítása a reprodukálhatóság érdekében.
    Befolyásolja a 'random', 'numpy' és 'os' modulokat.
    """
    random.seed(seed)
    np.random.seed(seed)
    # A Python hash-elést is fixáljuk (fontos lehet egyes adatszerkezeteknél)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed beállítva: {seed}")


# --- 2. Fájlkezelés (I/O) ---

def save_pickle(data: Any, filepath: Path):
    """
    Adat (pl. modell, scaler) mentése pickle fájlba.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.debug(f"Pickle fájl sikeresen mentve ide: {filepath}")
    except Exception as e:
        logging.error(f"Hiba a pickle fájl mentésekor ({filepath}): {e}")

def load_pickle(filepath: Path) -> Any:
    """
    Pickle fájl betöltése.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logging.debug(f"Pickle fájl sikeresen betöltve innen: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Hiba a pickle fájl betöltésekor ({filepath}): {e}")
        return None

def save_json(data: Dict, filepath: Path):
    """
    Szótár (pl. metrikák) mentése JSON fájlba.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.debug(f"JSON fájl sikeresen mentve ide: {filepath}")
    except Exception as e:
        logging.error(f"Hiba a JSON fájl mentésekor ({filepath}): {e}")

def load_json(filepath: Path) -> Dict:
    """
    JSON fájl betöltése szótárként.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"JSON fájl sikeresen betöltve innen: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Hiba a JSON fájl betöltésekor ({filepath}): {e}")
        return {}


# --- 3. Vizualizáció ---

def plot_metric_comparison(
    metrics_data: Dict[str, Dict[str, float]],
    metric_to_plot: str,
    save_path: Path
):
    """
    A benchmark eredményeiről (all_metrics) készít egy oszlopdiagramot 
    egy adott metrika alapján és elmenti.

    Args:
        metrics_data: A main.py által gyűjtött 'all_metrics' szótár.
                      Formátum: {'model_name': {'metric1': val1, 'metric2': val2}}
        metric_to_plot: Az a metrika, amit ábrázolni kell (pl. 'f1' vagy 'roc_auc').
        save_path: A .png fájl mentési útvonala.
    """
    logging.info(f"Összehasonlító ábra készítése a(z) '{metric_to_plot}' metrika alapján...")

    # Adatok előkészítése pandas DataFrame-be
    try:
        # Kinyerjük a modelleket és a kért metrika értékét
        plot_data = []
        for model_name, metrics in metrics_data.items():
            if 'error' in metrics:
                logging.warning(f"A(z) {model_name} modell hibára futott, kihagyva az ábráról.")
                continue
            
            value = metrics.get(metric_to_plot)
            if value is not None:
                plot_data.append({'Modell': model_name, metric_to_plot: value})
            else:
                logging.warning(f"A(z) {metric_to_plot} metrika hiányzik a(z) {model_name} modellnél.")

        if not plot_data:
            logging.error("Nincs adat az ábra készítéséhez.")
            return

        df = pd.DataFrame(plot_data)
        df = df.sort_values(by=metric_to_plot, ascending=False)

        # Ábra készítése (Seaborn-nal szebb)
        plt.figure(figsize=(10, 6))
        # ax = sns.barplot(
        #     x='Modell',
        #     y=metric_to_plot,
        #     data=df,
        #     palette='viridis' # Színpaletta
        # )
        
        # Címkék és címek
        ax.set_title(f"Modellek összehasonlítása: {metric_to_plot.upper()}", fontsize=16)
        ax.set_xlabel("Modell", fontsize=12)
        ax.set_ylabel(f"Pontszám ({metric_to_plot})", fontsize=12)
        plt.xticks(rotation=45, ha='right') # Forgatjuk a modellneveket, ha hosszúak

        # Értékek kiírása az oszlopok tetejére
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}", # 3 tizedesjegyre kerekítve
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )
        
        # Mentés
        plt.tight_layout() # Hogy minden beleférjen
        plt.savefig(save_path)
        plt.close() # Memóriakezelés miatt fontos bezárni
        
        logging.info(f"Összehasonlító ábra elmentve ide: {save_path}")

    except Exception as e:
        logging.error(f"Hiba az ábra készítése során: {e}")


# --- 4. Loggolás (Opcionális fejlettebb beállítás) ---

def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """
    Egy központi loggolási konfigurációt állít be.
    (A main.py-ban lévő egyszerűbb helyett használható)
    """
    level = logging.getLevelName(log_level.upper())
    
    # Alap formátum
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Alap logger beállítása
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Először eltávolítjuk a meglévő handlereket, hogy ne duplikálódjon
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # Konzol handler (StreamHandler)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    # Fájl handler (FileHandler), ha meg van adva
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    logging.info("Loggolás beállítva.")