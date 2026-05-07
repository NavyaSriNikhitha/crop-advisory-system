# ============================================================
# rag_pipeline.py
# What this file does:
#   1. Loads the ChromaDB knowledge base built by knowledge_base.py
#   2. Provides search_knowledge() function for semantic search
#   3. Implements RAG with fallback:
#      - If relevant docs found → return them for grounded answer
#      - If nothing relevant → signal to use Gemini general knowledge
#
# HOW RAG WORKS:
#   User Question → Embedding → Vector Search in ChromaDB
#   → Find similar documents → Pass to Gemini as context
#   → Gemini answers based on retrieved documents
#
# SIMILARITY THRESHOLD:
#   ChromaDB returns cosine distance (0 = identical, 2 = opposite)
#   We use threshold 0.7 — below this = relevant, above = not relevant
#   This is the "try/except" logic the user asked for:
#   if distance < threshold → use RAG
#   else → fall back to general Gemini knowledge
# ============================================================

import chromadb
from chromadb.utils import embedding_functions
import os


# ── Constants ─────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.5   # stricter — only use RAG for clearly relevant docs
TOP_K_RESULTS        = 3     # how many docs to retrieve
DB_PATH              = "vector_db"
COLLECTION_NAME      = "farming_knowledge"


# ── Load ChromaDB ─────────────────────────────────────────────
def load_knowledge_base():
    """
    Loads the ChromaDB collection from disk.
    Returns (collection, True) if successful, (None, False) if not built yet.
    """
    if not os.path.exists(DB_PATH):
        return None, False

    try:
        client = chromadb.PersistentClient(path=DB_PATH)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )

        count = collection.count()
        if count == 0:
            return None, False

        return collection, True

    except Exception as e:
        print(f"RAG load error: {e}")
        return None, False


# ── Search function ───────────────────────────────────────────
def search_knowledge(query: str, collection) -> dict:
    """
    Searches the knowledge base for documents relevant to the query.

    Returns a dict:
    {
        "use_rag": True/False,       # whether to use RAG or fallback
        "context": "...",            # retrieved documents as context string
        "sources": [...],            # list of source IDs found
        "best_distance": float,      # similarity score of best match
        "mode": "rag" / "general"    # which mode is being used
    }

    HOW THE TRY/ELSE LOGIC WORKS (what the user asked for):
    --------------------------------------------------------
    try:   query is related to knowledge base
           → search ChromaDB → distance < threshold
           → return relevant docs for grounded answer
    else:  query is NOT related to knowledge base
           → distance >= threshold → no relevant docs
           → fall back to Gemini's general farming knowledge
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K_RESULTS
        )

        distances  = results["distances"][0]
        documents  = results["documents"][0]
        ids        = results["ids"][0]
        metadatas  = results["metadatas"][0]

        best_distance = distances[0] if distances else 999

        # ── TRY: relevant documents found ────────────────────
        if best_distance < SIMILARITY_THRESHOLD:
            # Build context string from retrieved documents
            context_parts = []
            sources       = []

            for i, (doc, dist, doc_id, meta) in enumerate(
                zip(documents, distances, ids, metadatas)
            ):
                if dist < SIMILARITY_THRESHOLD:
                    context_parts.append(
                        f"[Source {i+1} — {meta.get('topic', doc_id)}]\n{doc}"
                    )
                    sources.append({
                        "id":       doc_id,
                        "topic":    meta.get("topic", ""),
                        "category": meta.get("category", ""),
                        "distance": round(dist, 4)
                    })

            context = "\n\n".join(context_parts)

            return {
                "use_rag":      True,
                "context":      context,
                "sources":      sources,
                "best_distance": round(best_distance, 4),
                "mode":         "rag"
            }

        # ── ELSE: no relevant documents — fall back ───────────
        else:
            return {
                "use_rag":      False,
                "context":      "",
                "sources":      [],
                "best_distance": round(best_distance, 4),
                "mode":         "general"
            }

    except Exception as e:
        # If search fails for any reason, fall back gracefully
        return {
            "use_rag":      False,
            "context":      "",
            "sources":      [],
            "best_distance": 999,
            "mode":         "general",
            "error":        str(e)
        }


# ── Build RAG prompt ──────────────────────────────────────────
def build_rag_prompt(query: str, context: str, system_prompt: str) -> str:
    """
    Builds the full prompt for Gemini when using RAG mode.
    Instructs Gemini to answer ONLY from the provided context.
    """
    return (
        f"{system_prompt}\n\n"
        f"=== RETRIEVED KNOWLEDGE BASE DOCUMENTS ===\n"
        f"{context}\n"
        f"=== END OF RETRIEVED DOCUMENTS ===\n\n"
        f"Using ONLY the information from the documents above, "
        f"answer the following question from the farmer. "
        f"If the documents don't fully cover the question, "
        f"supplement with your general farming knowledge but "
        f"prioritize the retrieved information.\n\n"
        f"Farmer's question: {query}"
    )


def build_general_prompt(query: str, system_prompt: str,
                         history_text: str = "") -> str:
    """
    Builds the full prompt for Gemini when using general knowledge mode.
    """
    return (
        f"{system_prompt}\n\n"
        f"Conversation so far:\n{history_text}\n"
        f"Farmer: {query}"
    )


# ── Test the pipeline (run directly to test) ──────────────────
if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    print("=" * 50)

    collection, loaded = load_knowledge_base()

    if not loaded:
        print("❌ Knowledge base not found!")
        print("Run: python knowledge_base.py first")
    else:
        print(f"✅ Knowledge base loaded — {collection.count()} documents")
        print()

        test_queries = [
            "what is drip irrigation and when should I use it?",
            "how to control aphids on cotton?",
            "what is the best crop for black soil?",
            "what is the weather like today?",   # should fall back to general
            "tell me a joke",                     # should fall back
        ]

        for query in test_queries:
            result = search_knowledge(query, collection)
            mode   = result["mode"].upper()
            dist   = result["best_distance"]
            print(f"Query: '{query[:50]}...' " if len(query) > 50 else f"Query: '{query}'")
            print(f"  Mode: {mode} | Best distance: {dist}")
            if result["use_rag"]:
                print(f"  Sources: {[s['topic'] for s in result['sources']]}")
            print()
