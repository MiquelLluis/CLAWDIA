import pickle

import yaml

from dataset import Dataset



with open('parameters.yaml') as f:
    cfg = yaml.safe_load(f)['dataset']

ds = Dataset(**cfg)
ds.setup_all()

with open('data/dataset.pkl', 'wb') as f:
    pickle.dump(ds, f)

print("DONETE")