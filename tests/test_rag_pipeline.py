from backend.rag_pipeline import RagPipeline

def test_rag_initial_state():
    rag = RagPipeline()
    assert rag.is_ready is False
