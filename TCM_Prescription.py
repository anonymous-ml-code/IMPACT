# ================================
# Import Libraries and Setup
# ================================
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, EncoderDecoderModel, AdamW
from tqdm import tqdm
import logging

# Set logging level to display errors only
logging.basicConfig(level=logging.ERROR)

# Set device: use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================
# Load and Prepare the Dataset
# ================================
# Load the CSV file.
# The CSV file should have at least these columns:
#   - MM_symptom: modern medical symptom (input)
#   - TCM_prescription: the corresponding TCM prescription (target)
df = pd.read_csv('TCM_Prescription.csv')

# (Optional) Check the columns
print("Columns in dataset:", df.columns.tolist())

# Split the data into training and validation sets (80-20 split)
train_size = 0.8
train_df = df.sample(frac=train_size, random_state=42)
val_df = df.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

print(f"FULL Dataset: {df.shape}")
print(f"TRAIN Dataset: {train_df.shape}")
print(f"VALIDATION Dataset: {val_df.shape}")


# ================================
# Define the Dataset Class
# ================================
class TCMPrescriptionDataset(Dataset):
    """
    PyTorch Dataset for TCM Prescription Generation.
    Each example consists of:
      - Input: MM_symptom text.
      - Target: TCM_prescription text.
    The texts are tokenized and padded/truncated to fixed lengths.
    """

    def __init__(self, dataframe, tokenizer, max_len_input, max_len_target):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input  # Maximum length for input (MM_symptom)
        self.max_len_target = max_len_target  # Maximum length for target (TCM_prescription)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve the MM_symptom and TCM_prescription from the dataframe
        mm_symptom = str(self.data.loc[index, 'MM_symptom'])
        tcm_prescription = str(self.data.loc[index, 'TCM_prescription'])

        # Tokenize the input text
        inputs = self.tokenizer(
            mm_symptom,
            max_length=self.max_len_input,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Tokenize the target text
        targets = self.tokenizer(
            tcm_prescription,
            max_length=self.max_len_target,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Squeeze to remove the extra dimension added by return_tensors
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()

        # Replace padding token id's in the labels by -100 so they are ignored by the loss function
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ================================
# Set Hyperparameters and Tokenizer
# ================================
MAX_LEN_INPUT = 128  # Maximum length for MM_symptom text
MAX_LEN_TARGET = 128  # Maximum length for TCM_prescription text
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5

# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# ================================
# Create Dataset and DataLoader Objects
# ================================
training_set = TCMPrescriptionDataset(train_df, tokenizer, MAX_LEN_INPUT, MAX_LEN_TARGET)
validation_set = TCMPrescriptionDataset(val_df, tokenizer, MAX_LEN_INPUT, MAX_LEN_TARGET)

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 2  # Adjust based on your system (or set to 0 on Windows)
}
val_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 2
}

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **val_params)

# ================================
# Initialize the Encoder-Decoder Model
# ================================
# Here, we use RoBERTa as both the encoder and the decoder.
# Note: For generation tasks, models such as BART or T5 are more common,
# but this example shows how to use RoBERTa in an encoder-decoder framework.
model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base")

# Set special token IDs for the decoder.
# Here we use the RoBERTa special tokens. Adjust if necessary.
if tokenizer.cls_token_id is not None:
    model.config.decoder_start_token_id = tokenizer.cls_token_id
else:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Set additional configuration for generation
model.config.max_length = MAX_LEN_TARGET
model.config.vocab_size = model.config.encoder.vocab_size

# Move the model to the selected device (GPU or CPU)
model.to(device)


def train_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass: the model computes the loss automatically when labels are provided.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


def generate_prescription(model, tokenizer, mm_symptom, max_length=128):
    """
    Generates a TCM prescription based on the provided MM_symptom.
    """
    model.eval()
    # Tokenize the input MM_symptom text
    inputs = tokenizer(
        mm_symptom,
        return_tensors="pt",
        max_length=MAX_LEN_INPUT,
        truncation=True,
        padding="max_length"
    ).to(device)

    # Generate output tokens using beam search
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=5,
        repetition_penalty=2.5,
        early_stopping=True
    )

    # Decode the generated tokens to a string (skipping special tokens)
    prescription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prescription


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for example in examples:
        # Tokenize text with explicit section separators
        text = "[CLS] " + " [SEP] ".join([
            " ".join(example.S1),
            " ".join(example.P1),
            " ".join(example.I1),
            " ".join(example.H1),
            " ".join(example.S2),
            " ".join(example.P2),
            # ... add more sections as needed
        ]) + " [SEP]"

        encoding = tokenizer(
            text,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        # Identify section boundaries
        sep_indices = [i for i, tok in enumerate(encoding.input_ids)
                       if tok == tokenizer.sep_token_id]
        sections = []
        prev = 0
        for sep in sep_indices:
            sections.append((prev, sep))
            prev = sep + 1

        # Generate custom attention mask
        seq_len = len(encoding.input_ids)
        attention_mask_2d = np.zeros((seq_len, seq_len), dtype=int)

        for i in range(seq_len):
            for j in range(seq_len):
                # Allow attending to [CLS] and [SEP]
                if encoding.input_ids[i] in {tokenizer.cls_token_id, tokenizer.sep_token_id}:
                    attention_mask_2d[i][j] = 1
                elif encoding.input_ids[j] in {tokenizer.cls_token_id, tokenizer.sep_token_id}:
                    attention_mask_2d[i][j] = 1
                else:
                    # Determine section relationships
                    i_sec = next((n for n, (s, e) in enumerate(sections) if s <= i <= e), -1)
                    j_sec = next((n for n, (s, e) in enumerate(sections) if s <= j <= e), -1)

                    # S-P-I-H activation rules
                    if i_sec in [1, 4]:  # S1/S2 sections
                        if j_sec in [i_sec, i_sec + 1, i_sec + 2, i_sec + 3]:
                            attention_mask_2d[i][j] = 1
                    elif i_sec in [2, 5]:  # P1/P2 sections
                        if j_sec in [i_sec - 1, i_sec]:
                            attention_mask_2d[i][j] = 1
                    elif i_sec in [3, 6]:  # I1/I2 sections
                        if j_sec in [i_sec - 2, i_sec]:
                            attention_mask_2d[i][j] = 1
                    elif i_sec in [4, 7]:  # H1/H2 sections
                        if j_sec in [i_sec - 3, i_sec]:
                            attention_mask_2d[i][j] = 1

        features.append({
            'input_ids': encoding.input_ids,
            'attention_mask_2d': attention_mask_2d,
            'labels': example.label
        })
    return features


class MaskedRoberta(RobertaForSequenceClassification):
    def forward(self, input_ids=None, attention_mask_2d=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask_2d  # Use our 2D mask
        )
        logits = self.classifier(outputs[0][:, 0, :])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)
        return logits


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.features[idx]['input_ids']),
            'attention_mask_2d': torch.tensor(self.features[idx]['attention_mask_2d']),
            'labels': torch.tensor(self.features[idx]['labels'])
        }

    def __len__(self):
        return len(self.features)


def main():

    # Load custom model
    model = MaskedRoberta.from_pretrained(args.bert_model, num_labels=num_labels)

    # Modified data preparation
    train_features = convert_examples_to_features(...)
    train_dataset = CustomDataset(train_features)

    # Training step modification
    for batch in train_dataloader:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask_2d'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, attention_mask_2d=masks, labels=labels)
        loss = outputs[0]
        loss.backward()

    # Example: Given an MM_symptom, generate a TCM prescription.
    test_mm_symptom = "Chronic fatigue, generalized weakness, pallor, and shortness of breath or tachycardia."
    generated_prescription = generate_prescription(model, tokenizer)
    print("\\nGenerated TCM Prescription:")
    print(generated_prescription)
