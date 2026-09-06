"""Métriques d'évaluation du RAG, sans appel LLM.

Deux étages mesurés séparément :
- la **recherche** — le passage attendu est-il parmi les k retournés ? (recall@k, MRR)
- la **réponse** — contient-elle les éléments attendus ? s'abstient-elle à bon escient ?

La vérité terrain de la recherche est une **citation exacte** du contrat
(`expected_evidence`) : un passage est correct s'il provient du bon fichier et
contient la citation. C'est plus robuste qu'un numéro de page ou un titre de section,
tous deux sensibles aux aléas du découpage.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"


def normaliser(texte: str) -> str:
    """Minuscules, sans accents, apostrophes droites, espaces réduits.

    Le texte extrait des PDF mélange tabulations, espaces insécables et apostrophes
    typographiques ; la comparaison doit y être insensible.
    """
    texte = texte.replace("’", "'").replace("`", "'")
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return " ".join(texte.lower().split())


@dataclass
class Question:
    id: str
    question: str
    category: str
    answerable: bool
    expected_evidence: list[dict] = field(default_factory=list)
    expected_section: str = ""
    expected_answer_contains: list[str] = field(default_factory=list)


def charger_questions(chemin: Path = QUESTIONS_PATH) -> list[Question]:
    questions = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            questions.append(Question(**json.loads(ligne)))
    return questions


# --- Recherche -------------------------------------------------------------------


def passage_contient(document: Document, preuve: dict) -> bool:
    """Vrai si le passage vient du bon fichier et contient l'une des citations."""
    if document.metadata.get("source") != preuve["source"]:
        return False
    contenu = normaliser(document.page_content)
    return any(normaliser(fragment) in contenu for fragment in preuve["any"])


def rangs_des_preuves(question: Question, passages: list[Document]) -> list[int | None]:
    """Pour chaque preuve attendue, le rang (1-indexé) du premier passage qui la porte."""
    rangs: list[int | None] = []
    for preuve in question.expected_evidence:
        rang = next((i + 1 for i, d in enumerate(passages) if passage_contient(d, preuve)), None)
        rangs.append(rang)
    return rangs


def recall_at_k(rangs: list[int | None], k: int) -> float:
    """Part des preuves attendues trouvées dans les k premiers passages.

    Vaut 1.0 pour une question factuelle à preuve unique retrouvée, et une fraction
    pour une question de synthèse dont seules certaines sources sont remontées.
    """
    if not rangs:
        return 0.0
    return sum(1 for r in rangs if r is not None and r <= k) / len(rangs)


def reciprocal_rank(rangs: list[int | None]) -> float:
    """1 / rang de la première preuve trouvée (0 si aucune)."""
    trouves = [r for r in rangs if r is not None]
    return 1.0 / min(trouves) if trouves else 0.0


# --- Réponse ---------------------------------------------------------------------


def reponse_contient_les_attendus(reponse: str, attendus: list[str]) -> bool:
    """Chaque attendu est une liste d'alternatives séparées par `|` : une suffit."""
    corps = normaliser(reponse)
    for attendu in attendus:
        alternatives = [normaliser(a) for a in attendu.split("|")]
        if not any(re.search(rf"(?<!\d){re.escape(a)}(?!\d)", corps) for a in alternatives):
            return False
    return True


@dataclass
class ScoreQuestion:
    id: str
    category: str
    answerable: bool
    rangs: list[int | None]
    recall_1: float
    recall_3: float
    recall_5: float
    rr: float
    # Étage réponse, renseigné seulement si une réponse a été générée.
    reponse: str | None = None
    abstention: bool | None = None
    juste: bool | None = None
    abstention_correcte: bool | None = None
    fidele: bool | None = None
    confiance: str | None = None
    tokens_entree: int = 0
    tokens_sortie: int = 0
    cout: float = 0.0


def scorer_recherche(question: Question, passages: list[Document]) -> ScoreQuestion:
    rangs = rangs_des_preuves(question, passages)
    return ScoreQuestion(
        id=question.id,
        category=question.category,
        answerable=question.answerable,
        rangs=rangs,
        recall_1=recall_at_k(rangs, 1),
        recall_3=recall_at_k(rangs, 3),
        recall_5=recall_at_k(rangs, 5),
        rr=reciprocal_rank(rangs),
    )


def scorer_reponse(
    score: ScoreQuestion, question: Question, reponse: str, abstention: bool
) -> None:
    """Complète un score de recherche avec le jugement de la réponse (règles, sans LLM)."""
    score.reponse = reponse
    score.abstention = abstention
    if question.answerable:
        score.juste = (not abstention) and reponse_contient_les_attendus(
            reponse, question.expected_answer_contains
        )
        score.abstention_correcte = None
    else:
        score.juste = None
        score.abstention_correcte = abstention


# --- Agrégation ------------------------------------------------------------------


def _moyenne(valeurs: list[float]) -> float | None:
    return sum(valeurs) / len(valeurs) if valeurs else None


def agreger(scores: list[ScoreQuestion]) -> dict:
    """Tableau de bord d'une configuration : recherche sur les questions à réponse,
    abstention sur les autres, coût sur l'ensemble."""
    avec_reponse = [s for s in scores if s.answerable]
    sans_reponse = [s for s in scores if not s.answerable]
    generes = [s for s in scores if s.reponse is not None]

    resume = {
        "n": len(scores),
        "recall@1": _moyenne([s.recall_1 for s in avec_reponse]),
        "recall@3": _moyenne([s.recall_3 for s in avec_reponse]),
        "recall@5": _moyenne([s.recall_5 for s in avec_reponse]),
        "mrr": _moyenne([s.rr for s in avec_reponse]),
    }
    if generes:
        justes = [s for s in avec_reponse if s.juste is not None]
        abstentions = [s for s in sans_reponse if s.abstention_correcte is not None]
        fideles = [s for s in generes if s.fidele is not None]
        fausses_abstentions = [s for s in avec_reponse if s.abstention]
        resume.update(
            {
                "justesse": _moyenne([float(s.juste) for s in justes]),
                "abstention_correcte": _moyenne(
                    [float(s.abstention_correcte) for s in abstentions]
                ),
                "fausses_abstentions": len(fausses_abstentions),
                "fidelite": _moyenne([float(s.fidele) for s in fideles]),
                "tokens_entree_moy": _moyenne([float(s.tokens_entree) for s in generes]),
                "tokens_sortie_moy": _moyenne([float(s.tokens_sortie) for s in generes]),
                "cout_total": sum(s.cout for s in generes),
                "cout_par_question": _moyenne([s.cout for s in generes]),
            }
        )
    return resume
