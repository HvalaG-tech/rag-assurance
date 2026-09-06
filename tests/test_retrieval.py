"""Tests de la recherche hybride vectoriel + BM25 (T2.2)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.retrieval import charger_corpus, fusion_rrf, rechercher, tokeniser

# --- Corpus de test : des passages courts et discriminants ---------------------

CORPUS = [
    Document(
        page_content="LA GARANTIE VOL\nLe vol est garanti après effraction, "
        "avec une franchise de 150 euros.",
        metadata={"source": "habitation.pdf", "page": 3, "section": "LA GARANTIE VOL"},
    ),
    Document(
        page_content="LA GARANTIE BRIS DE VITRES\nLes vitres brisées sont remplacées.",
        metadata={"source": "habitation.pdf", "page": 5, "section": "LA GARANTIE BRIS DE VITRES"},
    ),
    Document(
        page_content="LEXIQUE\nFranchise : somme restant à votre charge.",
        metadata={"source": "auto.pdf", "page": 1, "section": "LEXIQUE"},
    ),
    Document(
        page_content="ASSISTANCE\nLe remorquage est pris en charge.",
        metadata={"source": "auto.pdf", "page": 8, "section": "ASSISTANCE"},
    ),
]


class FauxVectorstore:
    """Double du vector store : renvoie un ordre fixé, sans modèle d'embeddings."""

    def __init__(self, ordre: list[Document], corpus: list[Document] | None = None):
        self.ordre = ordre
        self._corpus = corpus if corpus is not None else CORPUS
        self.derniers_kwargs: dict = {}

    def similarity_search(self, question: str, k: int = 4, **kwargs) -> list[Document]:
        self.derniers_kwargs = {"k": k, **kwargs}
        return self.ordre[:k]

    def get(self, include=None, **kwargs) -> dict:
        return {
            "ids": [str(i) for i in range(len(self._corpus))],
            "documents": [d.page_content for d in self._corpus],
            "metadatas": [d.metadata for d in self._corpus],
        }


# --- Tokenisation ---------------------------------------------------------------


def test_tokenise_en_minuscules_sans_ponctuation():
    assert tokeniser("La FRANCHISE est de 150 €, après effraction.") == [
        "la",
        "franchise",
        "est",
        "de",
        "150",
        "après",
        "effraction",
    ]


def test_tokenise_conserve_les_accents():
    """« résiliation » et « resiliation » ne doivent pas être confondus par erreur,
    mais surtout les termes accentués ne doivent pas être détruits."""
    assert "résiliation" in tokeniser("Conditions de RÉSILIATION du contrat")


# --- Fusion RRF -----------------------------------------------------------------


def test_rrf_privilegie_le_consensus_entre_les_deux_moteurs():
    """L'apport du RRF : un passage trouvé par les deux moteurs passe devant un
    passage que seul l'un des deux a vu, même mieux classé."""
    a, b, c = CORPUS[0], CORPUS[1], CORPUS[2]
    # `a` est 2e partout (2/62) ; `b` n'est 1er que d'un seul classement (1/61).
    fusionne = fusion_rrf([[b, a], [c, a]], k=60)

    assert fusionne[0] is a


@pytest.mark.parametrize("k", [1, 10, 60, 1000])
def test_rrf_favorise_les_rangs_contrastes_a_somme_egale(k: int):
    """Limite connue du RRF, vraie pour tout k : à somme de rangs égale, être
    1er puis 3e l'emporte sur être 2e deux fois — `1/(k+r)` est convexe, donc
    `1/(k+1) + 1/(k+3) > 2/(k+2)`. Aucun réglage de k n'inverse cela ; seule la
    présence dans les deux classements, elle, est réellement récompensée."""
    a, b, c = CORPUS[0], CORPUS[1], CORPUS[2]

    assert fusion_rrf([[b, a, c], [c, a, b]], k=k)[0] is not a


def test_rrf_applique_la_formule_attendue():
    a, b = CORPUS[0], CORPUS[1]
    resultats = fusion_rrf([[a, b], [a, b]], k=60, avec_scores=True)

    # a : rang 0 dans les deux -> 2 * 1/(60+1)
    assert resultats[0][1] == pytest.approx(2 / 61)
    assert resultats[1][1] == pytest.approx(2 / 62)


def test_rrf_deduplique_entre_classements():
    a, b = CORPUS[0], CORPUS[1]
    fusionne = fusion_rrf([[a, b], [b, a]], k=60)

    assert len(fusionne) == 2


def test_rrf_ignore_les_classements_vides():
    a = CORPUS[0]
    assert fusion_rrf([[a], []], k=60) == [a]
    assert fusion_rrf([[], []], k=60) == []


# --- Modes de recherche ---------------------------------------------------------


def test_mode_vector_n_utilise_que_le_vectoriel():
    ordre_vectoriel = [CORPUS[3], CORPUS[2]]
    vs = FauxVectorstore(ordre_vectoriel)

    resultats = rechercher("franchise vol", vs, Settings(retrieval_mode="vector", top_k=2))

    assert [d.page_content for d in resultats] == [d.page_content for d in ordre_vectoriel]
    # Sans BM25, le rang n'est renseigné que pour le moteur vectoriel.
    assert set(resultats[0].metadata["rangs"]) == {"vector"}


def test_mode_bm25_retrouve_le_terme_exact_que_le_vectoriel_manque():
    """Cas d'acceptation T2.2 : un terme rare et exact doit ressortir.

    Le vectoriel classe en tête des passages hors sujet ; BM25 s'appuie sur le mot.
    """
    vs = FauxVectorstore([CORPUS[3], CORPUS[2]])  # assistance, lexique : à côté

    resultats = rechercher("bris de vitres", vs, Settings(retrieval_mode="bm25", top_k=1))

    assert resultats[0].metadata["section"] == "LA GARANTIE BRIS DE VITRES"


def test_mode_hybride_repeche_ce_que_le_vectoriel_a_manque():
    """Le vectoriel place la bonne réponse en dernier ; l'hybride doit la remonter."""
    vs = FauxVectorstore([CORPUS[3], CORPUS[2], CORPUS[0], CORPUS[1]])

    hybride = rechercher("bris de vitres", vs, Settings(retrieval_mode="hybrid", top_k=2))
    vectoriel = rechercher("bris de vitres", vs, Settings(retrieval_mode="vector", top_k=2))

    sections_hybride = [d.metadata["section"] for d in hybride]
    sections_vectoriel = [d.metadata["section"] for d in vectoriel]
    assert "LA GARANTIE BRIS DE VITRES" in sections_hybride
    assert "LA GARANTIE BRIS DE VITRES" not in sections_vectoriel


def test_respecte_top_k():
    vs = FauxVectorstore(list(CORPUS))

    resultats = rechercher("franchise", vs, Settings(retrieval_mode="hybrid", top_k=2))

    assert len(resultats) == 2


def test_recupere_fetch_k_candidats_avant_fusion():
    vs = FauxVectorstore(list(CORPUS))

    rechercher("franchise", vs, Settings(retrieval_mode="hybrid", top_k=2, fetch_k=20))

    assert vs.derniers_kwargs["k"] == 20


def test_le_filtre_de_branche_est_transmis_au_vectoriel():
    vs = FauxVectorstore(list(CORPUS))

    rechercher("franchise", vs, Settings(retrieval_mode="vector"), filtre={"branche": "auto"})

    assert vs.derniers_kwargs["filter"] == {"branche": "auto"}


def test_mode_inconnu_est_rejete_a_la_configuration():
    with pytest.raises(ValueError, match="retrieval_mode"):
        Settings(retrieval_mode="magique")


# --- Chargement du corpus depuis Chroma -----------------------------------------


def test_charge_le_corpus_depuis_le_vectorstore():
    """BM25 se reconstruit depuis Chroma : pas de second index à synchroniser."""
    vs = FauxVectorstore([])

    corpus = charger_corpus(vs)

    assert len(corpus) == len(CORPUS)
    assert corpus[0].metadata["section"] == "LA GARANTIE VOL"


def test_corpus_vide_ne_fait_pas_echouer_la_recherche():
    vs = FauxVectorstore([], corpus=[])

    assert rechercher("franchise", vs, Settings(retrieval_mode="hybrid")) == []


# --- Reranking (T2.3) ------------------------------------------------------------


def _scorer_bris_de_vitres(paires):
    """Double du cross-encoder : note haut le passage qui parle de vitres."""
    return [3.0 if "vitres" in passage.lower() else -2.0 for _, passage in paires]


def test_le_reranker_remonte_le_bon_passage_en_tete():
    """Cas réel de T2.2 : l'hybride place « bris de vitres » 4e ; le reranker doit le
    remonter 1er à partir des mêmes candidats."""
    from app.retrieval import reranker

    candidats = [CORPUS[3], CORPUS[2], CORPUS[0], CORPUS[1]]  # bris de vitres en dernier

    reordonnes = reranker("bris de vitres", candidats, scorer=_scorer_bris_de_vitres)

    assert reordonnes[0][0].metadata["section"] == "LA GARANTIE BRIS DE VITRES"
    assert reordonnes[0][1] == 3.0


def test_reranker_sur_liste_vide():
    from app.retrieval import reranker

    assert reranker("x", [], scorer=_scorer_bris_de_vitres) == []


def test_recherche_avec_reranker_active():
    vs = FauxVectorstore([CORPUS[3], CORPUS[2], CORPUS[0], CORPUS[1]])
    cfg = Settings(retrieval_mode="hybrid", top_k=1, use_reranker=True)

    resultats = rechercher("bris de vitres", vs, cfg, scorer=_scorer_bris_de_vitres)

    assert resultats[0].metadata["section"] == "LA GARANTIE BRIS DE VITRES"
    # Le score affiché est celui du cross-encoder ; le score RRF reste disponible.
    assert resultats[0].metadata["score"] == 3.0
    assert 0 < resultats[0].metadata["score_rrf"] < 1


def test_les_resultats_portent_score_et_rangs_par_moteur():
    vs = FauxVectorstore([CORPUS[1], CORPUS[0]])

    resultats = rechercher("bris de vitres", vs, Settings(retrieval_mode="hybrid", top_k=2))

    premier = resultats[0].metadata
    assert 0 < premier["score"] <= 2 / 61
    assert "vector" in premier["rangs"] or "bm25" in premier["rangs"]
    assert all(isinstance(r, int) and r >= 1 for r in premier["rangs"].values())
