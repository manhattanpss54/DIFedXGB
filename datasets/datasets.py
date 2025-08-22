import pandas as pd
from urllib.request import urlretrieve
import os
from typing import Dict, Tuple

class DatasetDownloader:
    
    DATASET_INFO = {
        'agaricus': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
            'features': 22,
            'samples': 8124,
            'description': 'Mushroom classification dataset'
        },
        'weather': {
            'url': None,
            'features': 23,
            'samples': 106645,
            'description': 'Weather prediction dataset'
        },
        'santander': {
            'url': 'https://www.kaggle.com/c/santander-customer-satisfaction/data',
            'features': 369,
            'samples': 151836,
            'description': 'Customer satisfaction dataset'
        },
        'heart': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'features': 18,
            'samples': 319796,
            'description': 'Heart disease prediction dataset'
        },
        'susy': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00247/SUSY.csv.gz',
            'features': 19,
            'samples': 5000000,
            'description': 'Particle physics dataset'
        },
        'higgs': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
            'features': 29,
            'samples': 11000000,
            'description': 'Higgs boson detection dataset'
        }
    }

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def get_dataset_info(self, dataset_name: str) -> Dict:
        return self.DATASET_INFO.get(dataset_name.lower(), {})

    def download_dataset(self, dataset_name: str) -> Tuple[str, bool]:
        info = self.get_dataset_info(dataset_name)
        if not info:
            return (f"Dataset {dataset_name} not found", False)
        
        if not info['url']:
            return (f"No download URL available for {dataset_name}", False)
        
        try:
            filename = os.path.basename(info['url'])
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                urlretrieve(info['url'], filepath)
            
            return (filepath, True)
        except Exception as e:
            return (f"Error downloading {dataset_name}: {str(e)}", False)

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        filepath, success = self.download_dataset(dataset_name)
        if not success:
            raise ValueError(filepath)
        
        if filepath.endswith('.gz'):
            return pd.read_csv(filepath, compression='gzip')
        return pd.read_csv(filepath)