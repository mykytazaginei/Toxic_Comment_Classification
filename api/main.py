import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re
import spacy
import time
from contextlib import asynccontextmanager

loaded_model = None
vocab_to_int_map = None
SEQ_LENGTH_API = None
nlp_spacy = None
device_api = None
LABEL_COLUMNS_API = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
EMBEDDING_DIM_API = 100
OUTPUT_DIM_API = 6

class SimplerNBoWClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, padding_idx_val):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx_val)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text_batch):
        embedded = self.embedding(text_batch)
        averaged_embeddings = torch.mean(embedded, dim=1)
        logits = self.fc(averaged_embeddings)
        return logits

def preprocess_text_spacy_api(text, nlp_processor):
    if not nlp_processor:
        raise RuntimeError("spaCy model not loaded for API preprocessing.")
    if not isinstance(text, str): text = str(text)
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp_processor(text)
    processed_tokens = [
        token.lemma_ for token in doc if token.is_alpha and not token.is_stop
    ]
    return ' '.join(processed_tokens)

def numericalize_text_api(text, vocab_map):
    return [vocab_map.get(word, vocab_map.get('<unk>', 1)) for word in str(text).split() if word]

def pad_sequence_api(numerical_sequence, seq_len, pad_idx):
    if len(numerical_sequence) < seq_len:
        return numerical_sequence + [pad_idx] * (seq_len - len(numerical_sequence))
    elif len(numerical_sequence) > seq_len:
        return numerical_sequence[:seq_len]
    else:
        return numerical_sequence

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model, vocab_to_int_map, SEQ_LENGTH_API, nlp_spacy, device_api, EMBEDDING_DIM_API, OUTPUT_DIM_API
    print("Loading API resources...")
    start_load_time = time.time()

    device_api = torch.device("cpu")
    

    try:
        nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("spaCy 'en_core_web_sm' model loaded successfully for API.")
    except OSError:
        print("Critical Error: spaCy model 'en_core_web_sm' not found. API cannot start.")
        raise RuntimeError("spaCy model not found. Please run: python -m spacy download en_core_web_sm")

    vocab_config_path = '../model_artifacts/nbow_vocab_config.json'
    try:
        with open(vocab_config_path, 'r') as f:
            vocab_config = json.load(f)
        vocab_to_int_map = vocab_config['vocab_to_int']
        SEQ_LENGTH_API = vocab_config['SEQ_LENGTH']
        print(f"Vocabulary (size: {len(vocab_to_int_map)}) and SEQ_LENGTH ({SEQ_LENGTH_API}) loaded from {vocab_config_path}.")
    except FileNotFoundError:
        print(f"Critical Error: Vocabulary config file '{vocab_config_path}' not found. API cannot start.")
        raise RuntimeError(f"File not found: {vocab_config_path}")
    except KeyError:
        print(f"Critical Error: 'vocab_to_int' or 'SEQ_LENGTH' not found in '{vocab_config_path}'. API cannot start.")
        raise RuntimeError(f"Invalid format in: {vocab_config_path}")

    padding_idx_val_api = vocab_to_int_map.get('<pad>', 0)
    loaded_model = SimplerNBoWClassifier(len(vocab_to_int_map), EMBEDDING_DIM_API, OUTPUT_DIM_API, padding_idx_val_api)
    model_state_path = '../model_artifacts/nbow_model_state.pth'
    try:
        loaded_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
        loaded_model.to(device_api)
        loaded_model.eval()
        print(f"Model state loaded from '{model_state_path}' and moved to {device_api}.")
    except FileNotFoundError:
        print(f"Critical Error: Model state file '{model_state_path}' not found. API cannot start.")
        raise RuntimeError(f"File not found: {model_state_path}")
    except Exception as e:
        print(f"Critical Error loading model state: {e}. API cannot start.")
        raise RuntimeError(f"Error loading model state: {e}")

    end_load_time = time.time()
    print(f"API resources loaded in {end_load_time - start_load_time:.2f} seconds.")
    yield
    print("API shutting down...")

app = FastAPI(lifespan=lifespan)

class CommentInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label_probabilities: dict[str, float]

@app.post("/predict", response_model=PredictionOutput)
async def predict_toxicity(comment: CommentInput):
    if not all([loaded_model, vocab_to_int_map, SEQ_LENGTH_API, nlp_spacy]):
        raise HTTPException(status_code=503, detail="Model or resources not loaded. API not ready.")

    try:
        processed_text = preprocess_text_spacy_api(comment.text, nlp_spacy)
        numerical_sequence = numericalize_text_api(processed_text, vocab_to_int_map)
        padded_sequence = pad_sequence_api(numerical_sequence, SEQ_LENGTH_API, vocab_to_int_map.get('<pad>', 0))
        input_tensor = torch.LongTensor([padded_sequence]).to(device_api)
        
        with torch.no_grad():
            logits = loaded_model(input_tensor)
        
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        label_probs_dict = {LABEL_COLUMNS_API[i]: float(probabilities[i]) for i in range(len(LABEL_COLUMNS_API))}
        
        return PredictionOutput(label_probabilities=label_probs_dict)

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Toxicity Detection API is running!"}