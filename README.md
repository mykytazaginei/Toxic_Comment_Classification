# Toxicity Detector for Social Media Comments

## Overview
This project is focused on developing a system for detecting toxicity in social media comments across 6 different categories using Natural Language Processing (NLP). 
The system preprocesses input text, classifies it using a trained Neural Bag-of-Words (NBoW) model built with PyTorch, 
and provides access to this functionality via a RESTful web API and a simple web interface.

## Features
* Text preprocessing pipeline using spaCy (lemmatization, stopword removal, cleaning).
* Multi-label classification model based on Neural Bag-of-Words (NBoW) implemented in PyTorch.
* Model optimization for imbalanced datasets using a Weighted Loss Function.
* Classification Threshold Tuning to maximize F1-score for each toxicity category.
* RESTful API built with FastAPI to serve predictions.
* Simple interactive web UI built with Gradio for easy testing and demonstration.

## Technology Stack
* Language: Python 3
* NLP: spaCy
* ML & Data: PyTorch, scikit-learn (for metrics), pandas, numpy
* API: FastAPI, Uvicorn
* UI: Gradio

## Project Structure
```
Toxic_Comment_Classification/
├── .venv/                     # Virtual environment folder
├── api/                       # API code
│   └── main.py                # FastAPI application
├── gradio_app.py              # Gradio UI application
├── data/                      # Input data
│   └── train.csv              # Dataset from Jigsaw Challenge
├── notebooks/                 # Jupyter notebooks for development
│   └── Toxicity_Classification_NBoW.ipynb # Main notebook for training & evaluation
├── model_artifacts/           # Saved model artifacts
│   ├── nbow_model_state.pth   # Saved PyTorch model weights
│   └── nbow_vocab_config.json # Saved vocabulary and configuration
└── README.md                  # This file
```
## Dataset
The model was trained and evaluated using data from the Jigsaw Toxic Comment Classification Challenge dataset available on Kaggle. 
The specific data used for training should be placed in the data/ folder, named train.csv.

## Usage
The project consists of two main parts: the development notebook and the running application (API + UI).

### 1. Model Training & Evaluation (Jupyter Notebook)

You only need to run this notebook if you want to retrain the model or experiment with different parameters. The pre-trained artifacts are already provided in model_artifacts/.

Start Jupyter Notebook or Jupyter Lab from your project root directory.
Navigate to the notebooks/ folder and open Toxicity_Classification_NBoW.ipynb.
Run the cells sequentially. This notebook covers:
Data loading and preprocessing.
Building the vocabulary.
Defining the NBoW model in PyTorch.
Training the model using a weighted loss function.
Evaluating the model and performing threshold tuning.
Saving the model state (.pth) and vocabulary (.json) to the model_artifacts/ directory.

### 2. Running the API & UI

The application consists of two separate components that need to be run.

#### Step 1: Run the API Server

Run the Uvicorn server for the FastAPI application:

```Bash
uvicorn api.main:app --reload
```
The API will be available at http://127.0.0.1:8000. You can see the interactive API documentation at http://127.0.0.1:8000/docs.

#### Step 2: Run the Web UI

Open a second terminal in the project's root directory (leave the first one running the API). Ensure the virtual environment is activated.
Run the Gradio application script:

```Bash
python3 gradio_app.py
```

The interactive web UI will be available at the URL shown in the terminal, typically http://127.0.0.1:7860.
Now you can open the Gradio URL in your browser to test the toxicity detector visually.

## Results
The final model is a Neural Bag-of-Words classifier implemented in PyTorch. Its performance was significantly improved by using a weighted loss function to address class imbalance and by tuning classification thresholds to maximize the F1-score for each category.

The performance of the final model on the validation set is as follows:

Exact Match Ratio: ~89.8%. This represents the percentage of comments for which all 6 labels were predicted correctly.

F1-Scores per Category: The model demonstrates strong performance on more common classes while showing the expected challenges with rarer ones:

* obscene: 0.73
* toxic: 0.72
* insult: 0.67
* severe_toxic: 0.48
* identity_hate: 0.33
* threat: 0.30
* Overall F1-Score (Weighted Avg): 0.68

These results confirm that the chosen approach is effective for building a robust multi-label toxicity classifier. For detailed metrics including precision and recall for each class, please refer to the evaluation steps in the Jupyter Notebook.
