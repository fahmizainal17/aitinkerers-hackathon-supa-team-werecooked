import streamlit as st
import pandas as pd
import json
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
from component import page_style

page_style()

st.title("Model Benchmarking")

# Step 1: Select Model
tokenizer_options = [
    "wanadzhar913/malaysian-debertav2-finetune-on-boolq",
    "wanadzhar913/malaysian-mistral-llmasajudge-v2",
    "gpt-4-mini"  # Replace with actual identifier if available
]
selected_model = st.selectbox("Select a Model to Benchmark:", tokenizer_options)

# Load Model
tokenizer = AutoTokenizer.from_pretrained(selected_model)
model = AutoModelForSequenceClassification.from_pretrained(selected_model)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Step 2: Load Dataset
dataset_options = [
    "datasets/for_presentation/boolq-eng-val-200.jsonl",
    "datasets/for_presentation/boolq-malay-val-200.jsonl",
    "datasets/for_presentation/fib-eng-val-200.jsonl",
    "datasets/for_presentation/fib-malay-val-200.jsonl"
]
selected_dataset_path = st.selectbox("Select Dataset to Benchmark With:", dataset_options)

# Load dataset
def load_dataset(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

dataset = load_dataset(selected_dataset_path)
st.write(f"Loaded {len(dataset)} samples from the dataset.")

# Step 3: Run Benchmarking
def benchmark(pipe, dataset):
    accuracy = 0
    for data in dataset:
        result = pipe(data['question'], return_all_scores=False)
        label = result[0]['label']
        consistency = 1 if label == 'entailment' else 0
        if consistency == data['consistency']:
            accuracy += 1
    return accuracy / len(dataset)

if st.button("Run Benchmark"):
    with st.spinner('Benchmarking in progress...'):
        accuracy = benchmark(pipe, dataset)
    st.success(f'Benchmark completed. Model Accuracy: {accuracy:.2%}')
