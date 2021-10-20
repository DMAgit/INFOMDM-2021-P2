import os
import pandas as pd

df_training = pd.DataFrame(columns=['observation', 'label'])
df_testing = pd.DataFrame(columns=['observation', 'label'])

path_training_decepting = r'\data\training\decepting'
path_training_truthful = r'\data\training\truthful'

path_testing_decepting = r'\data\testing\decepting'
path_testing_truthful = r'\data\testing\truthful'

paths_training = [path_training_decepting, path_training_truthful]
paths_testing = [path_testing_decepting, path_testing_truthful]

for index, subpath in enumerate(paths_training):
    for directory in os.listdir(os.getcwd() + subpath):
        if os.path.isdir(os.getcwd() + subpath + '\\' + directory):
            for filename in os.listdir(os.getcwd() + subpath + '\\' + directory):
                with open(os.path.join(os.getcwd() + subpath + '\\' + directory, filename)) as f:
                    observation = f.read()
                    current_df = pd.DataFrame({'observation': [observation], 'label': [index % 2]})
                    df_training = df_training.append(current_df, ignore_index=True)

for index, subpath in enumerate(paths_testing):
    for directory in os.listdir(os.getcwd() + subpath):
        if os.path.isdir(os.getcwd() + subpath + '\\' + directory):
            for filename in os.listdir(os.getcwd() + subpath + '\\' + directory):
                with open(os.path.join(os.getcwd() + subpath + '\\' + directory, filename)) as f:
                    observation = f.read()
                    current_df = pd.DataFrame({'observation': [observation], 'label': [index % 2]})
                    df_testing = df_testing.append(current_df, ignore_index=True)

# print(df_training)
# print(df_testing)
