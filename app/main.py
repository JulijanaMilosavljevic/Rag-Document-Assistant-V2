import sys
import streamlit as st
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from backend.rag_pipeline import RagPipeline
from app.config import APP_TITLE, APP_DESCRIPTION


# ========== Helper: učitavanje CSS ==========
def load_css():
    css_path = Path("assets/styles/theme.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ========== Streamlit config ==========
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
    layout="wide",
)

load_css()
# ===========================
# SESSION STATE INIT
# ===========================
if "rag" not in st.session_state:
    st.session_state.rag = RagPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "last_files" not in st.session_state:
    st.session_state.last_files = []

if "auto_indexing" not in st.session_state:
    st.session_state.auto_indexing = False

rag = st.session_state.rag

# ========== Layout: Sidebar ==========
with st.sidebar:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path))

    st.markdown("### 📂 Upload PDF dokumenata")
    uploaded_files = st.file_uploader(
        "Izaberi jedan ili više PDF fajlova",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    top_k = st.slider(
        "Broj relevantnih pasusa",
        min_value=1,
        max_value=10,
        value=3,
    )
    st.markdown("---")
    st.caption("Diplomski rad – **Julijana Milosavljević** · 2025")


# ========== Header ==========

st.markdown(f"## 🔎 {APP_TITLE}")
st.markdown(APP_DESCRIPTION)

st.markdown("")

# ===========================
# DETEKCIJA PROMENE FAJLOVA
# ===========================
current_files = [f.name for f in uploaded_files] if uploaded_files else []

if set(current_files) != set(st.session_state.last_files):
    rag.reset()
    st.session_state.indexed = False
    st.session_state.auto_indexing = True
    st.session_state.last_files = current_files


# ===========================
# AUTO INDEXING
# ===========================
if uploaded_files and st.session_state.auto_indexing and not st.session_state.indexed:
    try:
        rag.build_index(uploaded_files)
        st.session_state.indexed = True
        st.session_state.auto_indexing = False
        st.success("Indeksiranje završeno!")
    except Exception as e:
        st.session_state.auto_indexing = False
        st.error(f"❌ Greška pri indeksiranju: {e}")
if not uploaded_files:
    st.info("📂 Uploaduj PDF dokument da započne indeksiranje.")
# ========== Main layout: 2 kolone ==========
left, right = st.columns([2, 1])

with left:
    with st.container():
        st.markdown("<div class='section-label'>Interakcija</div>", unsafe_allow_html=True)
        st.markdown("### 💬 Postavi pitanje dokumentima")
    def save_question():
     st.session_state["question_value"] = st.session_state["question_input"]
    question = st.text_input(
        "Unesi pitanje:",
        placeholder="Npr. \"O čemu se radi u ovom dokumentu?\"",
        key="question_input",
        on_change=save_question
    )

    ask_btn = st.button(
        "🚀 Pitaj AI",
        disabled=not st.session_state.indexed
    )


    if ask_btn:
            if not st.session_state.indexed or not rag.is_ready:
                st.error("❗ Indeks nije spreman. Uploaduj PDF dokument.")
                st.stop()

            if not question.strip():
                st.warning("Unesi pitanje.")
                st.stop()

            with st.spinner("🤔 Razmišljam..."):
                answer, sources , timing = rag.answer(question, top_k=top_k)

            # Answer box
            st.markdown(
                f"""
                <div class='answer-box fade-card'>
                    <div class='answer-title'>🧠 Odgovor</div>
                    {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sources
            st.markdown("### 📎 Relevantni pasusi")
            for s in sources:
                st.markdown(
                    f"""
                    <div class='source-card fade-card'>
                        <div class='source-title'>{s.get("title", "Dokument")}</div>
                        <div class='source-snippet'>{s.get("snippet", "")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with right:
    st.markdown("<div class='section-label'>Informacije</div>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Detalji o upitu")

    if st.session_state.indexed:
        st.success("Indeks je spreman ✓")
    else:
        st.info("Nema indeksa.Upload dokumente")

    st.markdown("#### 👀 Šta ova aplikacija radi?")
    st.markdown(
        """
        - 📄 PDF se učitava i pretvara u čist tekst
        - ✂️ Tekst se seče na logične blokove  
        - 🔤 Svaki blok dobija vektorsku reprezentaciju
        - 🧮 Sistem pravi memoriju nad dokumentima  
        - 🔍 Pronalazi najrelevantnije blokove na osnovu upita  
        - 🤖 LLM formira odgovor bez izmišljanja 
        """
    )

    st.markdown("#### ✨ Kako da dobiješ najbolji odgovor?")
    st.markdown(
        """
        - 🎯 Koristi precizne formulacije
        - 🧩 Pitaj o specifičnom delu dokumenta 
        - 📘 Traži strukturu: listu, tabelu, rezime 
        - 📑 Kombinuj pitanja o više dokumenata
        - 🔁 Postavljaj follow-up pitanja        
        """
    )
