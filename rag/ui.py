import os

import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit.log")

st.set_page_config(page_title="RAG Intelligence", layout="centered")
st.title("RAG Intelligence")
st.caption("Role-aware enterprise Q&A with citations and audit logging")

role = st.selectbox("Select your role", ["admin", "hr", "engineer", "finance", "intern"])
question = st.text_input("Ask a question")

if st.button("Ask") and question:
    with st.spinner("Querying..."):
        try:
            resp = requests.post(API_URL, json={"question": question, "role": role}, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.caption(f"**Citations:** {', '.join(data['citations']) if data['citations'] else 'None'}")
            st.caption(f"**Chunks used:** {data['chunks_used']} | **Denied sources:** {data['denied_sources']}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the FastAPI server is running on localhost:8000.")
        except Exception as e:
            st.error(f"Error: {e}")

with st.expander("Audit Trail (last 10 entries)"):
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r") as f:
            lines = f.readlines()
        for line in lines[-10:]:
            st.text(line.strip())
    else:
        st.info("No audit log entries yet.")
