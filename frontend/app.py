
# frontend/app.py
import os, requests, streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Docs", page_icon="🧪", layout="wide")

st.title("🧪 RAG documentaire")
st.caption("Posez une question sur les documents indexés. Les sources utilisées seront affichées.")

with st.form("ask-form"):
    q = st.text_input("Votre question", placeholder="Ex: Quelles sont les priorités du dossier 4_WAT ?", value="")
    top_k = st.slider("Nombre de passages à récupérer", 1, 10, 4)
    submitted = st.form_submit_button("Demander")

if submitted and q.strip():
    try:
        r = requests.post(f"{BACKEND_URL}/ask", json={"question": q, "top_k": top_k}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            st.subheader("Réponse")
            st.write(data.get("answer", ""))
            st.subheader("Sources")
            for s in data.get("sources", []):
                with st.expander(f"{s['file']} – score {s['score']:.3f}"):
                    st.write(s["chunk"])
        else:
            st.error(f"Erreur {r.status_code}: {r.text}")
    except Exception as e:
        st.error(str(e))

st.sidebar.header("État du service")
try:
    ping = requests.get(f"{BACKEND_URL}/healthcheck", timeout=5).json()
    st.sidebar.success("Backend OK ✅")
except Exception:
    st.sidebar.warning("Backend indisponible ⚠️")
