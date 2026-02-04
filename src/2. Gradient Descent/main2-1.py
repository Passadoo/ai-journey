import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Auto-detect device (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "meta-llama/Llama-3.2-1B"

print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)

print(inputs)

# Tokenizer prepends a speical <|begin_of_text|> token
print("\nInput tokens:")
for input_token_index in inputs["input_ids"].view(-1):
    print(input_token_index.item(), tokenizer.decode(input_token_index))

# Pass inputs into model
with torch.no_grad():
    outputs = model(inputs["input_ids"])

print(f"\nLogits shape: {outputs.logits.shape}")

# Get probabilities from logits using softmax
probabilities = F.softmax(outputs.logits, dim=-1)
print(f"Probabilities shape: {probabilities.shape}")

# Get the predicted token for the last position
last_token_logits = outputs.logits[0, -1, :]
last_token_probs = F.softmax(last_token_logits, dim=-1)
top_token_id = torch.argmax(last_token_probs).item()
predicted_token = tokenizer.decode([top_token_id])
print(f"\nPredicted next token: '{predicted_token}' (ID: {top_token_id})")
print(f"Probability: {last_token_probs[top_token_id].item():.4f}")

print()
print("Just look at final vector to see what text the model predicts next: \n")
# Just look at final vector to see what text the model predicts next
top_probs, top_indices = torch.topk(probabilities[0, -1, :], 10)
for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
    print(
        idx.item(), round(probabilities[0, -1, idx].item(), 5), tokenizer.decode([idx])
    )

# print("\nDecoding mystery tokens:")
# for token_id in [1304, 32671, 2130, 30743]:
#    print(f"Token {token_id}: '{tokenizer.decode([token_id])}' \n")

print()
print(
    "For comparison, here's the top results for the 4th position, these are the models predictions after The Capital of \n"
)
# For comparison, here's the top results for the 4th position,
# these are the models predictions after "The Capital of"
top_probs, top_indices = torch.topk(probabilities[0, 3, :], 10)
for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
    print(
        idx.item(), round(probabilities[0, 3, idx].item(), 5), tokenizer.decode([idx])
    )
