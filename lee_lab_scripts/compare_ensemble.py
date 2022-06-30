from hashlib import new
import pandas as pd

new = pd.read_csv('/data/anand/Automorph_data/biobank_results/M1/results_ensemble.csv')
old = pd.read_csv('/data/anand/Automorph_data/1000_kaggle_run/1000_kaggle_run_results/M1/results_ensemble.csv')

new['Name'] = new['Name'].apply(lambda x: x.split('/')[-1])
old['Name'] = old['Name'].apply(lambda x: x.split('/')[-1])

for _, data in new.iterrows():
    if data['Name'] in old['Name'].values:
        print(data.values)
        print(old.loc[old['Name'] == data['Name']])
        print('\n')