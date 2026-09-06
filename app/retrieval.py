"""Recherche hybride : vectoriel + BM25, fusionnés par Reciprocal Rank Fusion.

La recherche vectorielle seule échoue sur les termes exacts qui font le quotidien
de l'assurance — un montant de franchise, un nom de garantie, une référence de
contrat. Interrogé sur « bris de vitres », l'embedding rapproche volontiers
« dommages aux fenêtres » du bon passage… et aussi trois passages hors sujet.
BM25 apporte l'appariement lexical qui manque ; la fusion RRF combine les deux
classements sans avoir à calibrer des scores d'échelles différentes.

L'index BM25 est reconstruit depuis Chroma à la demande plutôt que persisté :
le corpus tient en mémoire, Chroma reste la source unique de vérité, et aucun
second index ne peut se désynchroniser du premier.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.config import Settings, settings

_MOTIF_JETON = re.compile(r"\w+", re.UNICODE)


def tokeniser(texte: str) -> list[str]:
    """Découpe un texte en jetons comparables : minuscules, accents conservés.

    Les accents sont significatifs en français ; les supprimer confondrait des
    termes distincts sans bénéfice de rappel notable sur ce corpus.
    """
    return _MOTIF_JETON.findall(texte.lower())


def charger_corpus(vectorstore: Any) -> list[Document]:
    """Relit tous les chunks indexés, pour alimenter l'index lexical."""
    donnees = vectorstore.get(include=["documents", "metadatas"])
    contenus = donnees.get("documents") or []
    metadonnees = donnees.get("metadatas") or [{}] * len(contenus)
    return [
        Document(page_content=contenu, metadata=dict(meta or {}))
        for contenu, meta in zip(contenus, metadonnees, strict=False)
    ]


def _cle(document: Document) -> tuple:
    """Identité d'un passage, pour le rapprocher entre deux classements."""
    meta = document.metadata
    return (meta.get("source"), meta.get("page"), document.page_content)


def fusion_rrf(
    classements: list[list[Document]],
    k: int = 60,
    avec_scores: bool = False,
) -> list[Document] | list[tuple[Document, float]]:
    """Fusionne plusieurs classements par Reciprocal Rank Fusion.

    Chaque passage marque `1 / (k + rang)` dans chaque classement où il figure,
    les rangs étant comptés à partir de 0. Seul le rang compte : les scores bruts
    du vectoriel et de BM25, non comparables, n'entrent jamais en jeu.
    """
    scores: dict[tuple, float] = {}
    documents: dict[tuple, Document] = {}

    for classement in classements:
        for rang, document in enumerate(classement):
            cle = _cle(document)
            scores[cle] = scores.get(cle, 0.0) + 1.0 / (k + rang + 1)
            documents.setdefault(cle, document)

    ordonnes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if avec_scores:
        return [(documents[cle], score) for cle, score in ordonnes]
    return [documents[cle] for cle, _ in ordonnes]


@lru_cache(maxsize=4)
def _index_bm25(empreinte: tuple[tuple, ...]) -> Any:
    """Construit l'index BM25. Mémoïsé : le corpus ne change qu'à la réindexation."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([list(jetons) for jetons in empreinte])


def _rechercher_bm25(corpus: list[Document], question: str, k: int) -> list[Document]:
    if not corpus:
        return []

    empreinte = tuple(tuple(tokeniser(d.page_content)) for d in corpus)
    index = _index_bm25(empreinte)
    scores = index.get_scores(tokeniser(question))

    meilleurs = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)[:k]
    # Un score nul signifie qu'aucun terme de la question n'apparaît : inutile de
    # faire remonter du bruit dans la fusion.
    return [corpus[i] for i in meilleurs if scores[i] > 0]


def _appliquer_filtre(corpus: list[Document], filtre: dict | None) -> list[Document]:
    """Reproduit sur BM25 le filtrage de métadonnées appliqué côté vectoriel."""
    if not filtre:
        return corpus
    return [d for d in corpus if all(d.metadata.get(c) == v for c, v in filtre.items())]


_RERANKERS: dict[str, Any] = {}


def _reranker(nom_modele: str) -> Any:
    """Cross-encoder chargé une fois par processus (20 s de chargement, PyTorch)."""
    if nom_modele not in _RERANKERS:
        from sentence_transformers import CrossEncoder

        _RERANKERS[nom_modele] = CrossEncoder(nom_modele)
    return _RERANKERS[nom_modele]


def reranker(
    question: str, candidats: list[Document], cfg: Settings = settings, scorer: Any = None
) -> list[tuple[Document, float]]:
    """Réordonne les candidats par pertinence réelle vis-à-vis de la question.

    Là où la fusion RRF ne voit que des rangs, le cross-encoder lit chaque paire
    (question, passage) et juge si le passage répond. C'est l'étape qui convertit
    le rappel de la fusion en précision. `scorer` permet d'injecter un double en test.
    """
    if not candidats:
        return []
    scorer = scorer or _reranker(cfg.reranker_model).predict
    scores = scorer([(question, d.page_content) for d in candidats])
    ordonnes = sorted(zip(candidats, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [(d, float(s)) for d, s in ordonnes]


def rechercher(
    question: str,
    vectorstore: Any,
    cfg: Settings = settings,
    filtre: dict | None = None,
    corpus: list[Document] | None = None,
    scorer: Any = None,
) -> list[Document]:
    """Retourne les `top_k` passages les plus pertinents selon le mode configuré.

    Chaîne complète : moteurs (vectoriel / BM25) → fusion RRF → reranking optionnel
    → `top_k`. Chaque passage retourné porte dans ses métadonnées un `score` (RRF,
    ou cross-encoder si activé) et le `rang` de chaque moteur qui l'a vu, pour que
    l'interface puisse expliquer d'où vient la pertinence.

    `corpus` permet d'injecter les chunks (tests, ou réutilisation entre appels) ;
    sinon ils sont relus depuis le vector store.
    """
    mode = cfg.retrieval_mode
    classements: list[list[Document]] = []
    noms: list[str] = []

    if mode in {"vector", "hybrid"}:
        kwargs: dict[str, Any] = {"k": cfg.fetch_k}
        if filtre:
            kwargs["filter"] = filtre
        classements.append(vectorstore.similarity_search(question, **kwargs))
        noms.append("vector")

    if mode in {"bm25", "hybrid"}:
        lexical = corpus if corpus is not None else charger_corpus(vectorstore)
        classements.append(
            _rechercher_bm25(_appliquer_filtre(lexical, filtre), question, cfg.fetch_k)
        )
        noms.append("bm25")

    scores = fusion_rrf(classements, k=cfg.rrf_k, avec_scores=True)
    if mode != "hybrid":
        # Un seul moteur : la « fusion » est un simple classement, le score RRF
        # reste utile comme mesure décroissante et bornée.
        scores = fusion_rrf(classements[:1], k=cfg.rrf_k, avec_scores=True)

    rangs: dict[tuple, dict[str, int]] = {}
    for nom, classement in zip(noms, classements, strict=True):
        for rang, document in enumerate(classement):
            rangs.setdefault(_cle(document), {})[nom] = rang + 1

    candidats: list[Document] = []
    for document, score in scores:
        meta = {**document.metadata, "score": score, "rangs": rangs.get(_cle(document), {})}
        candidats.append(Document(page_content=document.page_content, metadata=meta))

    if cfg.use_reranker:
        reordonnes = reranker(question, candidats[: cfg.fetch_k], cfg, scorer=scorer)
        candidats = []
        for document, score in reordonnes:
            document.metadata["score_rrf"] = document.metadata["score"]
            document.metadata["score"] = score
            candidats.append(document)

    return candidats[: cfg.top_k]
