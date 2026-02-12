import streamlit as st
import pandas as pd
import torch
import numpy as np
import shap
import pandas as pd
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
import altair as alt
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="OHS Risk Triage AI", layout="wide")

# Severity Classifications (from your notebook)
LABELS = ["None", "Minor", "Moderate", "Severe", "Major"]

# --- 1. LOAD MODEL (Cached for Performance) ---
@st.cache_resource
def load_model_pipeline():
    # REPLACE THIS with your actual Hugging Face model ID
    # Example: model_id = "stuart-42/ohs-severity-bert"
    model_id = "stuSterfc/ohs-severity-classifier" # Placeholder: SWAP THIS 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, return_dict=True)
    
    # Create a pipeline for SHAP compatibility
    pipe = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        top_k=None
    )
    return pipe

# Load resources
try:
    pipe = load_model_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- 2. THE GUARDRAILS (Logic from your Notebook) ---
def rules_based_alert(narrative, class_name):
    alerts = []
    
    # Needle Logic: Critical if keyword found but NOT predicted as Major
    needle_keywords = ['needle', 'sharp', 'bloodwork', 'lancet', 'suture']
    if any(keyword in narrative.lower() for keyword in needle_keywords) and class_name != "Major":
        alerts.append({
            "type": "CRITICAL",
            "title": "Needle/Sharp Detected",
            "body": "Model predicted non-Major, but needle keywords were found. Rare incidents of high incapacitation occur with needles. **Human review required.**"
        })

    # Middle Class Logic: Known overlap/confusion
    middle_classes = ["Minor", "Moderate", "Severe"]
    if class_name in middle_classes:
        alerts.append({
            "type": "WARNING",
            "title": "Middle Class Overlap",
            "body": f"Prediction is '{class_name}'. Your research shows high textual overlap between Minor/Moderate/Severe. **Review with care.**"
        })

    # Major Logic: High stakes regulation
    elif class_name == "Major":
        alerts.append({
            "type": "CRITICAL",
            "title": "MAJOR SEVERITY DETECTED",
            "body": "Due to high-stakes regulatory requirements (RIDDOR/OSHA), **mandatory human verification is required immediately** regardless of confidence score."
        })

    return alerts

# --- 3. HELPER: SHAP VISUALIZATION ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

# --- UI LAYOUT ---
st.title("🛡️ AI Health and Social Care Incident Triage")
st.warning(
    """
    **⚠️ LIMITATION OF LIABILITY**
    
    This application is an academic research prototype designed to demonstrate the potential of NLP in safety triage.
    
    * **Guidance Only:** The "Time Away From Work" prediction is a statistical estimate based on US OSHA data. It may not reflect specific UK clinical or legal outcomes.
    * **Human Oversight:** This tool **must not** replace professional judgment. Always verify predictions against official HSE guidance.
    * **Zero Warranty:** The author assumes no responsibility for errors, omissions, or actions taken based on this tool's output.
    """
)
# 2. Sub-header for Context/Author (Grey text, smaller)
st.caption("MSc Dissertation Project | Stuart Clark")

# 3. The "Why" (Bolded key terms for readability)
st.markdown(
    """
    **Objective:** Rapidly assess the potential **Time Away From Work** to prioritise investigation resources.
    """
)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/safety-hat.png", width=60)
    st.title("About")
    
    st.info(
        """
        **Research Prototype**
        Predicts **Incapacitation (Time Away from Work)** based on incident narratives.
        
        **Scope:**
        * **Sector:** Health & Social Care.
        * **Training Data:** US OSHA Records.
        * **Context:** Maps 'Days Away' patterns to UK RIDDOR reporting categories.
        """
    )
    
    st.markdown("### 📚 Project Resources")
    st.markdown("[📄 Methodology: Severity Definitions](https://drive.google.com/file/d/1honRpSK6RbSTckFkt15OUBADvoZJQRVS/view?usp=sharing)")
    st.markdown("[📊 Dissertation Abstract](https://drive.google.com/file/d/1o7QB1rlATQvhES7_HiNYZEJqZN3p0BHj/view?usp=sharing)")
    
    st.divider()
    st.caption("MSc Artificial Intelligence Project")
    st.markdown("Send feedback on LinkedIn: [Stuart Clark](https://www.linkedin.com/in/stuart-clark-161340164)")
    st.markdown("""
    <a href="https://www.buymeacoffee.com/stuart42" target="_blank">
        <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" >
    </a>
    """,
    unsafe_allow_html=True)

# Main Content Area

with st.container(border=True):
    st.subheader("1. Incident Narrative")
    
    narrative = st.text_area(
        "Describe the incident (Include: Who, What, Where, Injury Type, and Immediate Outcome.):",
        height=150,
        placeholder="E.g. On Friday at 14:00, a Care Assistant was helping a resident mobilize. The resident slipped, causing the staff member to twist their back. They visited A&E and have been signed off for 2 weeks...",
    )
    
    if narrative:
        word_count = len(narrative.split())
        st.caption(f"Word Count: {word_count}")
        if word_count < 10:
            st.warning("⚠️ Short description. Model accuracy improves with >20 words.")

    
    

if st.button("🔍 Analyze Incident", type="primary"):
    # 1. Feature Engineering (Matches your notebook's format)
    
    combined_text = f"{narrative}"
    col1, col2 = st.columns([2, 3])
    
    # --- MODEL PREDICTION ---
    # Run prediction
    outputs = pipe(combined_text)
    # Extract scores (pipeline returns a list of lists of dicts)
    scores = outputs[0] 
    # Find max score
    best_score = max(scores, key=lambda x: x['score'])
    predicted_label = best_score['label']
    
    # Map label to your specific bins if necessary (assuming model outputs LABEL_0, LABEL_1 etc.)
    # If your model already outputs "Major", "Minor", skip this mapping step.
    label_map = {"LABEL_0": "None", "LABEL_1": "Minor", "LABEL_2": "Moderate", "LABEL_3": "Severe", "LABEL_4": "Major"} 
    predicted_class = label_map.get(predicted_label, predicted_label)
    
    # Clean up the scores for the Chart as well    
    clean_scores = {}
    for s in scores:
        readable_name = label_map.get(s['label'], s['label'])
        clean_scores[readable_name] = s['score']  


    with col1:
        st.subheader("Assessment")
        
        # Display Prediction
        color_map = {"Major": "red", "Severe": "orange", "Moderate": "yellow", "Minor": "green", "None": "blue"}
        color = color_map.get(predicted_class, "grey")
        st.markdown(f":{color}[## {predicted_class}]")
        st.caption(f"Confidence: {best_score['score']:.2%}")
        if best_score['score'] < 0.60:
            st.warning("⚠️ **Low Confidence**\nThe model is unsure. Review carefully.")
        
        # Display Rules/Alerts
        alerts = rules_based_alert(combined_text, predicted_class)
        for alert in alerts:
            if alert['type'] == "CRITICAL":
                st.error(f"**{alert['title']}**: {alert['body']}")
            else:
                st.warning(f"**{alert['title']}**: {alert['body']}")

    with col2:
        st.subheader("Probability Distribution")
        
        # 1. Prepare Data
        class_order = ["None", "Minor", "Moderate", "Severe", "Major"]
        probs = [clean_scores.get(k, 0.0) for k in class_order]
        
        # Create a DataFrame (No index setting needed for Altair)
        df = pd.DataFrame({
            "Risk Class": class_order,
            "Probability": probs
        })
        
        # 2. Build Chart with Explicit Sort
        # We define the X axis and pass the 'class_order' list to 'sort'
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Risk Class', sort=class_order, title="Risk Severity"),
            y=alt.Y('Probability', title="Confidence Score"),
            tooltip=['Risk Class', alt.Tooltip('Probability', format='.1%')],
            color=alt.Color('Risk Class', scale={
                'domain': class_order,
                'range': ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#c0392b']  # Green to Red
            }, legend=None)
        ).properties(height=300)

        st.altair_chart(chart, use_container_width=True)

        # 3. Reference Index
        st.markdown("### ℹ️ Severity Reference")
        st.info("""
        | Risk Category | Time-Loss Criteria |
        | :--- | :--- |
        | **None** | 0 days (First aid) |
        | **Minor** | 1-2 days |
        | **Moderate** | 3-7 days |
        | **Severe** | 8-28 days (RIDDOR Reportable) |
        | **Major** | 29+ days (Long-term) |
        
        *Based on Classification Framework Figure 3.3*
        """)
            
            

    # --- SHAP EXPLANATION ---
    st.divider()
    st.subheader(f"3. AI Reasoning: Why '{predicted_class}'?")
    
    st.info(
        f"""
        **Interpretability Guide:**
        * 🟥 **RED BARS (Evidence For):** Words that pushed the model **TOWARDS** predicting **{predicted_class}**.
        * 🟦 **BLUE BARS (Evidence Against):** Words that pushed the model **AWAY** from this prediction.
        
        Key words are correlated to the severity class based on patterns learned from the training data. 
        Use this to understand the model's reasoning and identify any potential incident causes for further investigation.
        """
    )
    
    
    with st.spinner("Calculating SHAP values..."):
        # We use a Permutation explainer as in your notebook, but optimized for display
        
        # 2. SETUP EXPL AINER
        # We pass the wrapper function, NOT the raw pipe
        masker = shap.maskers.Text(r"\s+")
        
        def model_predictor(texts):
            # FIXED: Convert numpy array to python list to prevent tokenizer crash
            if hasattr(texts, "tolist"):
                texts = texts.tolist()
            
            # Get the list of dicts from the pipeline
            outputs = pipe(texts)
            
            # Strip the keys, keep only the scores
            return np.array([[s['score'] for s in out] for out in outputs])
        
        explainer = shap.Explainer(model_predictor, masker=masker, algorithm="permutation")
        
        # 3. CALCULATE
        shap_values = explainer([combined_text])
        
        # 4. PICK THE TARGET CLASS
        # We need the integer index (0=None ... 4=Major) to pick the right bar in the graph
        class_order = ["None", "Minor", "Moderate", "Severe", "Major"]
        class_index = class_order.index(predicted_class) 

        # 5. DRAW
        # shap_values[0] is the text. [:, class_index] slices just the target class.
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0][:, class_index], max_display=14, show=False)
        plt.title(f"Why '{predicted_class}'? (Key Drivers)")
        
        st.pyplot(fig)
        

