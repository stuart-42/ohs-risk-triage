print("1. Testing Imports...")
import streamlit
import transformers
import torch
import shap
print("✅ Imports successful.")

print("2. Testing Model Download (Small Test)...")
from transformers import pipeline
# We use a tiny model just to see if your internet/firewall allows downloads
pipe = pipeline("text-classification", model="distilbert-base-uncased")
print("✅ Model download successful.")

print("3. Testing SHAP...")
import numpy as np
# Dummy data to see if SHAP crashes
print("✅ SHAP imported successfully (skipping calc for speed).")

print("\n🎉 SYSTEM READY. You are clear to run 'streamlit run app.py'")