"""Tests unitaires du pipeline RAG (sans appel LLM ni API externe)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.config import Settings
from app.embeddings import load_documents, split_documents
from app.rag_chain import format_context


def test_load_documents_lit_les_txt(tmp_path: Path):
    (tmp_path / "cg_incendie.txt").write_text(
        "Exclusion : dommages causés intentionnellement.", encoding="utf-8"
    )
    (tmp_path / "ignore.csv").write_text("a,b", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "cg_incendie.txt"


def test_split_documents_respecte_chunk_size():
    cfg = Settings(chunk_size=50, chunk_overlap=10)
    doc = Document(page_content="phrase. " * 60, metadata={"source": "x.txt"})

    chunks = split_documents([doc], cfg)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 60 for c in chunks)
    # Les métadonnées de provenance doivent survivre au découpage.
    assert all(c.metadata["source"] == "x.txt" for c in chunks)


def test_format_context_expose_source_et_page():
    docs = [
        Document(page_content="Délai de 5 jours.", metadata={"source": "cg.pdf", "page": 2}),
        Document(page_content="Franchise 150 €.", metadata={"source": "faq.txt"}),
    ]

    context = format_context(docs)

    # page est 0-indexée côté PyPDFLoader, affichée en 1-indexé.
    assert "Source: cg.pdf, p.3" in context
    assert "Source: faq.txt" in context
    assert "Franchise 150 €." in context


def test_format_context_identifie_l_assureur_et_la_section():
    """Le modèle doit savoir de quel contrat vient l'extrait, pas seulement de quel fichier."""
    docs = [
        Document(
            page_content="Vol : 2 jours ouvrés.",
            metadata={
                "source": "dg_allsecur_auto.pdf",
                "page": 29,
                "assureur": "allianz",
                "branche": "auto",
                "section": "DISPOSITIONS EN CAS DE SINISTRE",
            },
        )
    ]

    context = format_context(docs)

    assert "Source: dg_allsecur_auto.pdf, p.30" in context
    assert "assureur : ALLIANZ" in context
    assert "branche : auto" in context
    assert "section : DISPOSITIONS EN CAS DE SINISTRE" in context
