"""Pré-calcule les réponses du parcours de démonstration (`data/demo_answers.json`).

À relancer après toute modification du corpus, du prompt ou de la recherche : les
réponses servies en mode démo doivent refléter le système réel.

Usage : python -m scripts.build_demo_answers        (requiert une clé API)
"""

from __future__ import annotations

import sys
from dataclasses import replace

from app.config import settings
from app.demo import DEMO_ANSWERS_PATH, PARCOURS, sauvegarder_reponses_demo
from app.rag_chain import answer


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Meilleure configuration mesurée en évaluation : hybride + reranker. Le
    # reranker est trop lourd pour l'hébergement gratuit, mais ici il tourne en
    # local, une fois — les visiteurs profitent de sa précision sans en payer la RAM.
    cfg = replace(settings, use_reranker=True, retrieval_mode="hybrid")

    reponses = {}
    total = 0.0
    for etape in PARCOURS:
        question = etape["question"]
        print(f"· {etape['etiquette']}  {question[:70]}…", flush=True)
        reponse = answer(question, cfg)
        reponses[question] = reponse
        total += reponse.cout_estime
        print(
            f"  → {'abstention' if reponse.abstention else 'réponse'} · "
            f"confiance {reponse.confiance} · {reponse.cout_estime:.4f} $"
        )

    sauvegarder_reponses_demo(reponses)
    print(f"\n{len(reponses)} réponses écrites dans {DEMO_ANSWERS_PATH} (coût total {total:.3f} $)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
