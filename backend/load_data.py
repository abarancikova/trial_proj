import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

#load the dataset
dataset = load_dataset('backend/dataset/arg_quality_rank_30k.csv')

#split the dataset into features and target variable
X = dataset['argument']
y = dataset['WA']

#split the dataset into 80% train and 20% valid
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




