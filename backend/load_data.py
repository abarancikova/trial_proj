import torch
from transformers import BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from transformers import AdamW
import pandas as pd

# Load dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Tokenize input texts
def prepare_data(X_train, X_val, y_train, y_val):
    # Tokenize text features using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='pt')
    X_val_tokens = tokenizer(list(X_val), padding=True, truncation=True, return_tensors='pt')

    # Standardize the target variable
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_tokens['input_ids'], dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_tokens['input_ids'], dtype=torch.long)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    # Create PyTorch DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tokens['attention_mask'], y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, X_val_tokens['attention_mask'], y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader

# Fine-tune BERT model for regression
def fine_tune_bert_model(train_loader, val_loader):
    # Load pre-trained BERT model for classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}')

    # Return the fine-tuned model
    return model

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset('backend/dataset/arg_quality_rank_30k.csv')

    # Split the dataset into features and target variable
    X = dataset['argument']
    y = dataset['WA']

    # Split the dataset into 80% train and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare data loaders
    train_loader, val_loader = prepare_data(X_train, X_val, y_train, y_val)

    # Fine-tune BERT model for regression
    model = fine_tune_bert_model(train_loader, val_loader)

    # Save the fine-tuned model
    model.save_pretrained('fine_tuned_model2')

