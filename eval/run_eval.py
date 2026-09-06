"""Harnais d'évaluation : une commande, un tableau de métriques.

    python -m eval.run_eval --compare            # recherche seule, toutes configurations
    python -m eval.run_eval --llm                # + génération sur la configuration retenue
    python -m eval.run_eval --llm --judge        # + juge LLM de fidélité
    python -m eval.run_eval --llm --limit 5      # essai à blanc sur 5 questions

La recherche ne coûte rien et tourne sur toutes les configurations. La génération
coûte un appel par question : elle ne tourne que sur la configuration de référence
(`CONFIG_REFERENCE`), sauf `--llm-all`.

Sorties : `eval/results/<date>_<tag>.md` (lisible) et `.json` (comparable).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from app.config import CHROMA_DIR, Settings, settings
from app.embeddings import get_vectorstore, ingest
from app.rag_chain import answer, format_context
from app.retrieval import charger_corpus, rechercher
from eval.metrics import (
    ScoreQuestion,
    agreger,
    charger_questions,
    scorer_recherche,
    scorer_reponse,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Index d'évaluation, séparés de l'index de démonstration : la comparaison des
# stratégies de découpage exige un index par stratégie.
EVAL_CHROMA_DIR = CHROMA_DIR.parent / "chroma_eval"

# Nom -> surcharges de Settings. Chaque axe est isolé pour chiffrer sa contribution.
CONFIGURATIONS: dict[str, dict] = {
    "fixed · vector": {"chunking_strategy": "fixed", "retrieval_mode": "vector"},
    "fixed · hybrid": {"chunking_strategy": "fixed", "retrieval_mode": "hybrid"},
    "semantic · vector": {"chunking_strategy": "semantic", "retrieval_mode": "vector"},
    "semantic · bm25": {"chunking_strategy": "semantic", "retrieval_mode": "bm25"},
    "semantic · hybrid": {"chunking_strategy": "semantic", "retrieval_mode": "hybrid"},
    "semantic · hybrid · rerank": {
        "chunking_strategy": "semantic",
        "retrieval_mode": "hybrid",
        "use_reranker": True,
    },
}
CONFIG_REFERENCE = "semantic · hybrid · rerank"


def configuration(nom: str, base: Settings = settings) -> Settings:
    surcharges = CONFIGURATIONS[nom]
    strategie = surcharges.get("chunking_strategy", base.chunking_strategy)
    return replace(
        base,
        **surcharges,
        collection_name=f"eval_{strategie}",
        persist_directory=str(EVAL_CHROMA_DIR),
    )


def preparer_index(cfg: Settings) -> None:
    """Construit l'index d'évaluation de la stratégie si absent (une fois par stratégie)."""
    store = get_vectorstore(cfg)
    if store.get(include=[])["ids"]:
        return
    print(f"  · construction de l'index {cfg.collection_name}…", flush=True)
    n = ingest(cfg=cfg)
    print(f"    {n} chunks")


def evaluer(nom: str, questions, avec_llm: bool, avec_juge: bool) -> list[ScoreQuestion]:
    cfg = configuration(nom)
    preparer_index(cfg)
    store = get_vectorstore(cfg)
    corpus = charger_corpus(store)

    scores: list[ScoreQuestion] = []
    for q in questions:
        # 20 candidats pour mesurer le rang jusqu'à 20 ; les métriques @k tronquent.
        cfg_large = replace(cfg, top_k=20)
        passages = rechercher(q.question, store, cfg_large, corpus=corpus)
        score = scorer_recherche(q, passages)

        if avec_llm:
            sources = passages[: cfg.top_k]
            reponse = answer(q.question, cfg, sources=sources)
            scorer_reponse(score, q, reponse.answer, reponse.abstention)
            score.confiance = reponse.confiance
            score.tokens_entree = reponse.usage.get("input_tokens", 0)
            score.tokens_sortie = reponse.usage.get("output_tokens", 0)
            score.cout = reponse.cout_estime
            if avec_juge:
                from eval.judge import juger_fidelite

                score.fidele, _ = juger_fidelite(
                    q.question, format_context(sources), reponse.answer, cfg
                )
        scores.append(score)
        marque = "✓" if score.recall_5 == 1 or not q.answerable else "✗"
        print(f"  {marque} {q.id} r@5={score.recall_5:.0%}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return scores


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.0%}"


def tableau_markdown(resultats: dict[str, dict]) -> str:
    colonnes = ["recall@1", "recall@3", "recall@5", "mrr"]
    avec_llm = any("justesse" in r for r in resultats.values())
    if avec_llm:
        colonnes += [
            "justesse",
            "abstention_correcte",
            "fausses_abstentions",
            "fidelite",
            "cout_par_question",
        ]

    entete = "| Configuration | " + " | ".join(colonnes) + " |"
    sep = "|---|" + "|".join("---:" for _ in colonnes) + "|"
    lignes = [entete, sep]
    for nom, r in resultats.items():
        cellules = []
        for c in colonnes:
            v = r.get(c)
            if c == "fausses_abstentions":
                cellules.append("—" if v is None else str(v))
            elif c == "cout_par_question":
                cellules.append("—" if v is None else f"{v:.4f} $")
            elif c == "mrr":
                cellules.append("—" if v is None else f"{v:.2f}")
            else:
                cellules.append(_pct(v))
        lignes.append(f"| {nom} | " + " | ".join(cellules) + " |")
    return "\n".join(lignes)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parseur.add_argument(
        "--compare", action="store_true", help="toutes les configurations (recherche)"
    )
    parseur.add_argument("--llm", action="store_true", help="générer les réponses (coût API)")
    parseur.add_argument(
        "--llm-all", action="store_true", help="générer sur toutes les configurations"
    )
    parseur.add_argument("--judge", action="store_true", help="juge LLM de fidélité (coût API)")
    parseur.add_argument("--limit", type=int, default=0, help="n premières questions seulement")
    parseur.add_argument("--tag", default="", help="suffixe du fichier de résultats")
    args = parseur.parse_args()

    questions = charger_questions()
    if args.limit:
        questions = questions[: args.limit]
    noms = list(CONFIGURATIONS) if (args.compare or args.llm_all) else [CONFIG_REFERENCE]

    resultats: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    debut = time.perf_counter()
    for nom in noms:
        avec_llm = args.llm_all or (args.llm and nom == CONFIG_REFERENCE)
        print(f"▶ {nom}" + (" (+ génération)" if avec_llm else ""))
        scores = evaluer(nom, questions, avec_llm, args.judge and avec_llm)
        resultats[nom] = agreger(scores)
        details[nom] = [asdict(s) for s in scores]

    duree = time.perf_counter() - debut
    horodatage = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR.mkdir(exist_ok=True)

    md = [
        f"# Évaluation — {horodatage}",
        "",
        f"{len(questions)} questions · {duree:.0f} s · "
        f"configuration de référence : `{CONFIG_REFERENCE}`",
        "",
        tableau_markdown(resultats),
        "",
        "recall@k et MRR : questions à réponse dans le corpus (part des preuves attendues"
        " retrouvées dans les k premiers passages). justesse : réponses générées contenant"
        " les éléments attendus. abstention_correcte : questions hors corpus sur lesquelles"
        " le système a déclaré l'information absente. fausses_abstentions : questions à"
        " réponse sur lesquelles il s'est abstenu à tort.",
    ]
    (RESULTS_DIR / f"{horodatage}{tag}.md").write_text("\n".join(md), encoding="utf-8")
    if args.llm and not args.limit:
        # Copie lue par l'interface (« Comment ce système est évalué »).
        (RESULTS_DIR / "latest.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS_DIR / f"{horodatage}{tag}.json").write_text(
        json.dumps({"resume": resultats, "details": details}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print()
    print(tableau_markdown(resultats))
    print(f"\n→ eval/results/{horodatage}{tag}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
