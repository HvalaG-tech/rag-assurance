"""Test d'acceptation T2.2 sur le corpus réel indexé.

Ces tests exigent un index construit (`python -m app.embeddings`) et chargent le
modèle d'embeddings : ils sont ignorés si l'index est absent, pour que la suite
reste exécutable sur un dépôt fraîchement cloné.
"""

from __future__ import annotations

import pytest

from app.config import Settings, settings

pytest.importorskip("rank_bm25")


@pytest.fixture(scope="module")
def index():
    """Vector store réel et corpus lexical, ou skip si rien n'est indexé."""
    from app.embeddings import get_vectorstore
    from app.retrieval import charger_corpus

    vectorstore = get_vectorstore()
    corpus = charger_corpus(vectorstore)
    if len(corpus) < 100:
        pytest.skip("index absent ou incomplet — lancer `python -m app.embeddings`")
    return vectorstore, corpus


def _sections(resultats) -> list[str]:
    return [d.metadata.get("section", "") for d in resultats]


@pytest.mark.parametrize(
    ("question", "attendu"),
    [
        ("bris de vitres", "BRIS DE VITRES"),
        ("garantie vol après effraction", "VOL"),
    ],
)
def test_l_hybride_retrouve_un_terme_exact_que_le_vectoriel_manque(index, question, attendu):
    """Critère d'acceptation T2.2.

    L'exemple « que dit l'article 7 ? » de la roadmap initiale n'est pas utilisable :
    le corpus ne comporte pas d'articles numérotés (cf. `docs/CORPUS.md`). Ces deux
    questions portent sur des noms de garantie réellement présents.
    """
    from app.retrieval import rechercher

    vectorstore, corpus = index

    hybride = rechercher(
        question, vectorstore, Settings(retrieval_mode="hybrid", top_k=5), corpus=corpus
    )
    vectoriel = rechercher(
        question, vectorstore, Settings(retrieval_mode="vector", top_k=5), corpus=corpus
    )

    assert any(attendu in s for s in _sections(hybride)), f"hybride : {_sections(hybride)}"
    assert not any(attendu in s for s in _sections(vectoriel)), (
        "le vectoriel seul ne devrait pas trouver ce passage"
    )


def test_le_filtre_de_branche_restreint_bien_les_resultats(index):
    from app.retrieval import rechercher

    vectorstore, corpus = index

    resultats = rechercher(
        "franchise", vectorstore, settings, filtre={"branche": "sante"}, corpus=corpus
    )

    assert resultats
    assert all(d.metadata["branche"] == "sante" for d in resultats)
