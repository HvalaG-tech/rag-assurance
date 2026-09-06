"""Construit l'index de démonstration versionné dans le dépôt (`data/chroma/`).

Principe : aucun embedding n'est calculé au démarrage de la démo en ligne. L'index
est construit ici, en local, puis versionné — l'application le trouve prêt.

Usage : python -m scripts.build_index
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.config import CHROMA_DIR, settings
from app.embeddings import get_vectorstore, ingest

TAILLE_MAX_MO = 20


def taille_mo(dossier: Path) -> float:
    return sum(f.stat().st_size for f in dossier.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(
        f"Index -> {CHROMA_DIR}\n"
        f"  embeddings : {settings.embedding_provider} / {settings.hf_embedding_model}\n"
        f"  découpage  : {settings.chunking_strategy} "
        f"({settings.chunk_size}/{settings.chunk_overlap})"
    )
    n = ingest(cfg=settings, reset=True)
    total = len(get_vectorstore(settings).get(include=[])["ids"])
    taille = taille_mo(CHROMA_DIR)
    print(f"  {n} chunks indexés ({total} dans la collection) · {taille:.1f} Mo")

    if taille > TAILLE_MAX_MO:
        print(f"  ⚠ l'index dépasse {TAILLE_MAX_MO} Mo : réduire le corpus avant de versionner.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
