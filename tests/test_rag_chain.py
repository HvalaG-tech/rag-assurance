"""Tests de la chaîne de réponse : parsing, abstention, confiance (sans appel LLM)."""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import MARQUEUR_ABSTENTION, Settings
from app.rag_chain import (
    _extraire_confiance,
    combiner_confiance,
    confiance_recherche,
    est_une_abstention,
    texte_de_la_reponse,
)


def test_extrait_le_texte_des_blocs_de_reponse():
    """Claude Opus 5 renvoie des blocs (thinking + text) : seul le texte compte."""
    blocs = [
        {"type": "thinking", "thinking": "", "signature": "abc"},
        {"type": "text", "text": "Le délai est de 2 jours ouvrés."},
    ]
    assert texte_de_la_reponse(blocs) == "Le délai est de 2 jours ouvrés."
    assert texte_de_la_reponse("simple") == "simple"


def test_separe_la_ligne_de_confiance_du_corps():
    corps, niveau = _extraire_confiance("Réponse sourcée.\n\nCONFIANCE: élevée")
    assert corps == "Réponse sourcée."
    assert niveau == "élevée"


def test_confiance_absente_vaut_moyenne():
    corps, niveau = _extraire_confiance("Réponse sans ligne finale.")
    assert corps == "Réponse sans ligne finale."
    assert niveau == "moyenne"


def test_le_marqueur_d_abstention_est_detectable_en_tete():
    corps, _ = _extraire_confiance(
        f"{MARQUEUR_ABSTENTION} Le montant figure aux CP.\nCONFIANCE: faible"
    )
    assert corps.upper().startswith(MARQUEUR_ABSTENTION.rstrip("."))


def test_confiance_globale_retient_le_signal_le_plus_prudent():
    assert combiner_confiance("élevée", "élevée") == "élevée"
    assert combiner_confiance("élevée", "faible") == "faible"
    assert combiner_confiance("moyenne", "élevée") == "moyenne"


def test_confiance_recherche_normalise_le_score_rrf():
    cfg = Settings()
    plafond = 2 / (cfg.rrf_k + 1)  # 1er des deux moteurs
    assert (
        confiance_recherche([Document(page_content="x", metadata={"score": plafond})], cfg)
        == "élevée"
    )
    assert (
        confiance_recherche([Document(page_content="x", metadata={"score": plafond * 0.1})], cfg)
        == "faible"
    )
    assert confiance_recherche([], cfg) == "faible"


def test_confiance_recherche_avec_reranker_passe_par_une_sigmoide():
    cfg = Settings()
    fort = Document(page_content="x", metadata={"score": 3.0, "score_rrf": 0.01})
    faible = Document(page_content="x", metadata={"score": -4.0, "score_rrf": 0.01})
    assert confiance_recherche([fort], cfg) == "élevée"
    assert confiance_recherche([faible], cfg) == "faible"


def test_abstention_detectee_malgre_la_mise_en_forme_markdown():
    """Le modèle met le marqueur en gras ou en titre : la détection doit tenir."""
    assert est_une_abstention(f"**{MARQUEUR_ABSTENTION}**\n\nRien dans les extraits.")
    assert est_une_abstention(f"### {MARQUEUR_ABSTENTION} …")
    assert est_une_abstention(f"{MARQUEUR_ABSTENTION} Le montant figure aux CP.")
    assert not est_une_abstention("Le délai est de 2 jours ouvrés [Source: x, p.3].")
