"""Juge LLM de fidélité : la réponse s'appuie-t-elle sur le contexte fourni ?

Les métriques par règles (`metrics.py`) vérifient qu'une réponse contient les bons
éléments ; elles ne détectent pas une affirmation juste mais **inventée** — un montant
correct par hasard, ou tiré des connaissances du modèle plutôt que du contrat. Le juge
lit la réponse phrase par phrase face aux extraits et tranche.

Coûteux (un appel par question) : activé explicitement par `--judge`.
"""

from __future__ import annotations

from dataclasses import replace

from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings, settings
from app.rag_chain import get_llm

PROMPT_JUGE = """Tu es un auditeur. On te donne des EXTRAITS de conditions générales \
d'assurance, une QUESTION et la RÉPONSE d'un assistant qui n'avait le droit d'utiliser \
que ces extraits.

Vérifie chaque affirmation de la RÉPONSE : est-elle appuyée par les EXTRAITS ?
- Une réponse qui déclare l'information absente est FIDÈLE si les extraits ne la contiennent \
effectivement pas.
- Une réponse est NON FIDÈLE dès qu'une affirmation factuelle (montant, délai, condition, \
exclusion) n'apparaît dans aucun extrait, même si elle est plausible ou vraie par ailleurs.
- Les reformulations et les synthèses d'éléments présents sont fidèles.

Réponds sur deux lignes exactement :
VERDICT: FIDELE ou NON_FIDELE
RAISON: une phrase.

EXTRAITS :
{contexte}

QUESTION : {question}

RÉPONSE :
{reponse}"""


def juger_fidelite(
    question: str, contexte: str, reponse: str, cfg: Settings = settings
) -> tuple[bool, str]:
    """Retourne (fidèle, raison). Le juge tourne à effort minimal : c'est une lecture."""
    cfg_juge = replace(cfg, effort="low", max_tokens=400)
    chaine = ChatPromptTemplate.from_messages([("human", PROMPT_JUGE)]) | get_llm(cfg_juge)
    sortie = chaine.invoke({"contexte": contexte, "question": question, "reponse": reponse})
    texte = sortie.content if isinstance(sortie.content, str) else str(sortie.content)

    verdict = "NON_FIDELE" not in texte.upper() and "FIDELE" in texte.upper()
    raison = next(
        (ligne.split(":", 1)[1].strip() for ligne in texte.splitlines() if "RAISON" in ligne),
        texte.strip()[:200],
    )
    return verdict, raison
