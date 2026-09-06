"""Parcours de démonstration guidé et réponses pré-calculées.

Six questions choisies pour raconter une histoire dans l'ordre : factuelle simple,
terme exact (l'apport de la recherche hybride), synthèse multi-documents, question
hors corpus (l'abstention), comparaison entre deux contrats, piège sur une exclusion.

En mode démo, leurs réponses sont servies depuis `data/demo_answers.json` : un
prospect voit le système fonctionner sans clé et sans coût.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.documents import Document

from app.config import ROOT
from app.rag_chain import RagAnswer

DEMO_ANSWERS_PATH = ROOT / "data" / "demo_answers.json"

PARCOURS: list[dict[str, str]] = [
    {
        "etiquette": "🔎 Factuelle",
        "question": "Dans quel délai dois-je déclarer le vol de mon véhicule chez Allianz ?",
        "pourquoi": "Une question simple : un délai précis, dans un seul contrat.",
    },
    {
        "etiquette": "🎯 Terme exact",
        "question": "Que couvre la garantie bris de vitres du contrat habitation MMA ?",
        "pourquoi": "Un nom de garantie exact : c'est là que la recherche lexicale (BM25) "
        "rattrape ce que la recherche vectorielle seule manque.",
    },
    {
        "etiquette": "🧩 Synthèse",
        "question": "Quel est le délai de prescription des actions dérivant du contrat dans les "
        "contrats Allianz auto, MMA habitation et SMACL RC ?",
        "pourquoi": "Trois documents à croiser pour une seule réponse.",
    },
    {
        "etiquette": "🚫 Hors corpus",
        "question": "Quel capital est versé aux bénéficiaires en cas de décès dans le contrat "
        "d'assurance vie ?",
        "pourquoi": "Aucun document ne répond. Le système doit le dire, pas inventer.",
    },
    {
        "etiquette": "⚖️ Comparaison",
        "question": "Compare les délais de déclaration d'un vol chez Allianz (auto) et chez "
        "MMA (habitation).",
        "pourquoi": "Deux contrats, deux assureurs, une réponse structurée.",
    },
    {
        "etiquette": "⚠️ Exclusion",
        "question": "Suis-je couvert par le contrat auto Allianz si je conduis en état d'ivresse ?",
        "pourquoi": "Question piège : la réponse est dans les exclusions, pas dans les garanties.",
    },
]


def serialiser(reponse: RagAnswer) -> dict:
    """RagAnswer -> dict JSON (les Document deviennent {page_content, metadata})."""
    donnees = asdict(reponse)
    donnees["sources"] = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in reponse.sources
    ]
    return donnees


def deserialiser(donnees: dict) -> RagAnswer:
    sources = [Document(**d) for d in donnees.get("sources", [])]
    return RagAnswer(**{**donnees, "sources": sources})


def charger_reponses_demo(chemin: Path = DEMO_ANSWERS_PATH) -> dict[str, RagAnswer]:
    """Réponses pré-calculées, indexées par question. Vide si le fichier manque."""
    if not chemin.exists():
        return {}
    brut = json.loads(chemin.read_text(encoding="utf-8"))
    return {question: deserialiser(d) for question, d in brut.items()}


def sauvegarder_reponses_demo(
    reponses: dict[str, RagAnswer], chemin: Path = DEMO_ANSWERS_PATH
) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps({q: serialiser(r) for q, r in reponses.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
