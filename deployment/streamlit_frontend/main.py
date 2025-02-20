import json
import time
import requests

import streamlit as st

# API details
API_URL = "http://localhost:5001/v1/generateText"
HEADERS = {"Content-Type": "application/json"}

# Streamlit UI
st.title("Team werecooked's LLM-as-a-Judge Web App")

# Sidebar for inputs
st.sidebar.header("Input Data")
passage = st.sidebar.text_area("Passage", "Enter the passage here...")
summary = st.sidebar.text_area("Summary", "Enter the summary here...")

# Button to submit
if st.button("Generate Critique"):
    if passage and summary:
        data = {"passage": passage, "question": summary}
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            llm_response = response.json()
            st.subheader("LLM Response")
            st.write(llm_response)
            st.write(f"Latency: {latency:.2f} seconds")
        else:
            st.error("Error calling API")
    else:
        st.warning("Please enter both a passage and a summary.")
