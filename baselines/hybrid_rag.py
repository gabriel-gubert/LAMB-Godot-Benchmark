"""Baseline 3: Advanced Hybrid Code-to-Doc RAG.

Pipeline stages:
  LAMB Unique Chunk Loading -> AST Extraction -> HyDE -> BM25 + Dense RRF -> Cross-Encoder -> Generation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, RRF_K, TOP_K_RERANK
from instances import chat_model, embeddings_model

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Prompts
# ──────────────────────────────────────────────────────────────────────

_HYDE_SYSTEM = """\
You are a technical documentation author for the Godot Engine.

Given a list of GDScript AST symbols from a Godot 3.x script, write a concise \
migration guide describing the equivalent Godot 4.x classes, methods, signals, \
and properties for each symbol. Focus on API renames, removed features, and \
new patterns."""

_HYDE_USER = """\
Godot 3.x AST Symbols:
{symbols}

Write a Godot 4.x migration guide covering every symbol above."""

_GEN_SYSTEM = """\
You are an expert GDScript developer. You MUST migrate the \
provided Godot 3.x script to valid Godot 4.x GDScript using ONLY the \
retrieved documentation context below. Do NOT rely on any prior knowledge \
outside this context.

--- BEGIN Retrieved Context ---
{context}
--- END Retrieved Context ---

Return ONLY valid Godot 4.x GDScript code."""

_GEN_USER = """\
Convert the following Godot 3.x GDScript file to Godot 4.x \
using ONLY the retrieved context provided in the system message.

--- BEGIN Godot 3.x GDScript ---
{source_code}
--- END Godot 3.x GDScript ---

Return ONLY the converted Godot 4.x GDScript code."""


# ──────────────────────────────────────────────────────────────────────
#  LAMB Chunk Loader (Exact Unbundled Unique Chunks Ingestion)
# ──────────────────────────────────────────────────────────────────────

class LambChunkLoader:
    """Loads the exact unbundled unique chunk set exported by LAMB."""

    @classmethod
    def load_from_json(cls, json_path: Path) -> List[Document]:
        """Ingests LAMB's unique chunk collection directly from a JSON file.

        Supports list of objects with 'hash', 'text', 'version', and 'sources'.
        """
        if not json_path.exists():
            logger.error("LAMB chunk file not found at %s", json_path)
            return []

        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as err:
            logger.error("Failed to parse LAMB chunks JSON at %s: %s", json_path, err)
            return []

        documents: List[Document] = []

        for idx, item in enumerate(data):
            if isinstance(item, str):
                text_content = item.strip()
                metadata = {"chunk_id": idx, "source": "lamb_unique_chunks"}
            elif isinstance(item, dict):
                text_content = item.get("text", item.get("page_content", "")).strip()
                metadata = {
                    "hash": item.get("hash", str(idx)),
                    "version": item.get("version", "v2"),
                    "sources": item.get("sources", []),
                }
            else:
                continue

            if not text_content:
                continue

            # Context enrichment using sources if available
            sources = metadata.get("sources", [])
            source_header = f"Godot 4 API Reference ({', '.join(sources)})\n\n" if sources else ""
            enriched_text = f"{source_header}{text_content}"

            documents.append(
                Document(
                    page_content=enriched_text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Successfully loaded %d unique LAMB chunks into RAG Document objects.",
            len(documents),
        )
        return documents


# ──────────────────────────────────────────────────────────────────────
#  Hybrid RAG Pipeline
# ──────────────────────────────────────────────────────────────────────

class HybridRAGMigrator:
    """Hybrid RAG migrator combining dense FAISS + sparse BM25 retrieval with Cross-Encoder re-ranking."""

    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        model=None,
        embed_model=None,
    ):
        self.model = model or chat_model
        self.embed_model = embed_model or embeddings_model
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self._docs: List[Document] = docs or []

        # Lazy-built indices
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._doc_embeddings: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None

        if docs:
            self.index_documents(docs)

    # ---------------------------------------------------------------- #
    #  Index construction
    # ---------------------------------------------------------------- #

    def index_documents(self, docs: List[Document]) -> None:
        """Build FAISS and BM25 indices over *docs*."""
        self._docs = docs
        logger.info("Indexing %d documents ...", len(docs))

        # Dense – FAISS
        texts = [d.page_content for d in docs]
        embeddings = self.embed_model.embed_documents(texts)
        self._doc_embeddings = np.array(embeddings, dtype="float32")
        dim = self._doc_embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self._doc_embeddings)
        self._faiss_index.add(self._doc_embeddings)

        # Sparse – BM25
        tokenised = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenised)

        logger.info("Indices built (dim=%d).", dim)

    # ---------------------------------------------------------------- #
    #  AST feature extraction (tree-sitter)
    # ---------------------------------------------------------------- #

    @staticmethod
    def extract_ast_symbols(source_code: str) -> List[str]:
        """Extract structural symbols from GDScript source.

        Uses tree-sitter with the GDScript grammar when available; falls
        back to regex-based extraction otherwise.
        """
        symbols: List[str] = []
        try:
            import tree_sitter_language_pack as tslp

            parser = tslp.get_parser("gdscript")
            tree = parser.parse(source_code.encode())
            HybridRAGMigrator._walk_tree(tree.root_node, symbols)
        except Exception:
            # Fallback: regex extraction
            import re

            symbols += re.findall(r"\bclass_name\s+(\w+)", source_code)
            symbols += re.findall(r"\bextends\s+(\w+)", source_code)
            symbols += re.findall(r"\bfunc\s+(\w+)", source_code)
            symbols += re.findall(r"\bsignal\s+(\w+)", source_code)
            symbols += re.findall(r"\bvar\s+(\w+)", source_code)
            symbols += re.findall(r"\.(\w+)\s*\(", source_code)

        return list(dict.fromkeys(symbols))

    @staticmethod
    def _walk_tree(node, symbols: List[str]) -> None:
        """Recursively collect identifier-like nodes from the AST."""
        if node.type in ("name", "identifier", "type", "attribute"):
            text = node.text.decode()
            if text not in symbols:
                symbols.append(text)
        for child in node.children:
            HybridRAGMigrator._walk_tree(child, symbols)

    # ---------------------------------------------------------------- #
    #  HyDE generation
    # ---------------------------------------------------------------- #

    def generate_hyde(self, symbols: List[str]) -> str:
        """Generate a synthetic migration guide from AST symbols."""
        symbols_str = ", ".join(symbols)
        messages = [
            SystemMessage(content=_HYDE_SYSTEM),
            HumanMessage(content=_HYDE_USER.format(symbols=symbols_str)),
        ]
        response = self.model.invoke(messages)
        return response.content.strip()

    # ---------------------------------------------------------------- #
    #  Hybrid retrieval (BM25 + Dense + RRF)
    # ---------------------------------------------------------------- #

    def hybrid_search(
        self,
        query_dense: str,
        query_sparse_tokens: List[str],
        top_n: int = 20,
    ) -> List[Tuple[Document, float]]:
        """Return top-*n* candidates via Reciprocal Rank Fusion."""
        if not self._faiss_index or not self._bm25:
            raise RuntimeError("Indices not built. Call index_documents() first.")

        # Dense scores
        q_emb = np.array(
            self.embed_model.embed_query(query_dense), dtype="float32"
        ).reshape(1, -1)
        faiss.normalize_L2(q_emb)
        scores_dense, indices_dense = self._faiss_index.search(
            q_emb, min(top_n * 2, len(self._docs))
        )

        # Sparse scores
        bm25_scores = self._bm25.get_scores(
            [t.lower() for t in query_sparse_tokens]
        )
        indices_sparse = np.argsort(bm25_scores)[::-1][: top_n * 2]

        # Reciprocal Rank Fusion
        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(indices_dense[0]):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(indices_sparse):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(self._docs[idx], score) for idx, score in ranked]

    # ---------------------------------------------------------------- #
    #  Cross-encoder re-ranking
    # ---------------------------------------------------------------- #

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = TOP_K_RERANK,
    ) -> List[Document]:
        """Re-rank candidates with a cross-encoder and return top-*k*."""
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for (doc, _rrf), _ce in scored[:top_k]]

    # ---------------------------------------------------------------- #
    #  End-to-end migration
    # ---------------------------------------------------------------- #

    def migrate_file(
        self,
        legacy_path: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Full RAG migration pipeline for a single file."""
        source_code = legacy_path.read_text(errors="replace")

        t0 = time.perf_counter()

        # 1. AST extraction
        symbols = self.extract_ast_symbols(source_code)
        logger.info("Extracted %d symbols from %s", len(symbols), legacy_path.name)

        # 2. HyDE
        hyde_text = self.generate_hyde(symbols)

        # 3. Hybrid search
        candidates = self.hybrid_search(
            query_dense=hyde_text,
            query_sparse_tokens=symbols,
        )

        # 4. Cross-encoder re-ranking
        top_docs = self.rerank(query=hyde_text, candidates=candidates)

        # Extract hashes/metadata from retrieved context documents
        retrieved_chunk_hashes = [
            d.metadata.get("hash", "unknown") for d in top_docs
        ]

        # 5. Non-parametric generation
        context = "\n\n---\n\n".join(d.page_content for d in top_docs)
        messages = [
            SystemMessage(content=_GEN_SYSTEM.format(context=context)),
            HumanMessage(content=_GEN_USER.format(source_code=source_code)),
        ]
        response = self.model.invoke(messages)
        generated_code = response.content.strip()

        # Strip markdown fences
        generated_code = _strip_code_fences(generated_code)

        latency = time.perf_counter() - t0

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated_code)
            logger.info("Written RAG migrated file to %s", output_path)

        token_usage = {}
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})

        return {
            "generated_code": generated_code,
            "latency": latency,
            "tokens": token_usage,
            "symbols_extracted": symbols,
            "num_context_chunks": len(top_docs),
            "retrieved_chunk_hashes": retrieved_chunk_hashes,
        }


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```gdscript ... ```)."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)