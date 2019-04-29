import pandas as pd
data_file = 'data/quora/train.csv'
data_file2 = 'data/quora/test.csv'

data = pd.read_csv(data_file)
data2 = pd.read_csv(data_file2)

for dataset in [data, data2]:
    for index, row in dataset.iterrows():
        for question in ['question1','question2']:
            for word in row[question].split():
                print(word)

