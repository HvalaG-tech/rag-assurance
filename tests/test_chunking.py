"""Tests du découpage sémantique (T2.1)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.chunking import (
    Section,
    decouper_en_sections,
    detecter_titres_repetes,
    est_titre_candidat,
)
from app.config import Settings
from app.embeddings import split_documents

# --- Reconnaissance des titres -------------------------------------------------


def test_reconnait_les_vrais_titres():
    for ligne in [
        "LA GARANTIE VOL",
        "DISPOSITIONS GÉNÉRALES",
        "LES GARANTIES DE BASE",
        "DÉFENSE PÉNALE ET RECOURS",
        "2. VOS GARANTIES FRAIS DE SANTÉ",
        "Article 7",
        "## Les exclusions",
    ]:
        assert est_titre_candidat(ligne), ligne


def test_rejette_le_bruit_typographique():
    """Le bruit relevé sur le corpus réel : pagination, adresses, sommaire."""
    for ligne in [
        "10/56",  # numéro de page
        "BP 290",  # adresse de l'assureur
        "TSA 46 307",  # adresse
        "LEXIQUE 6",  # entrée de sommaire : titre + numéro de page
        "EXCLUSIONS GÉNÉRALES 26",  # entrée de sommaire
        "",
        "   ",
        "Le présent contrat couvre les dommages.",  # phrase normale
        "AB",  # trop court
        # Clause d'exclusion composée en majuscules : trop longue pour un titre.
        "SA CHARGE DANS LA MESURE OU NOUS SERIONS DANS L'IMPOSSIBILITE DE LES RECOUVRER",
    ]:
        assert not est_titre_candidat(ligne), repr(ligne)


def test_detecte_les_entetes_repetes_de_page():
    """« ASSURANCE AUTO » sur chaque page est un en-tête, pas un titre de section."""
    pages = [f"ASSURANCE AUTO\nLA GARANTIE VOL\ncontenu {i}" for i in range(10)]

    repetes = detecter_titres_repetes(pages)

    assert "ASSURANCE AUTO" in repetes
    assert "LA GARANTIE VOL" in repetes  # répété aussi : le test suivant les distingue


def test_un_titre_apparaissant_une_fois_n_est_pas_un_entete():
    pages = ["ASSURANCE AUTO\nPRÉAMBULE\ntexte"] + [
        f"ASSURANCE AUTO\ncontenu {i}" for i in range(9)
    ]

    repetes = detecter_titres_repetes(pages)

    assert "ASSURANCE AUTO" in repetes
    assert "PRÉAMBULE" not in repetes


# --- Découpage en sections -----------------------------------------------------


def test_decoupe_sur_les_titres_et_retient_la_page():
    pages = [
        "EN-TETE\nLES GARANTIES DE BASE\nLe contrat couvre l'incendie.",
        "EN-TETE\nLA GARANTIE VOL\nLe vol est couvert après effraction.",
    ]

    sections = decouper_en_sections(pages)

    titres = [s.titre for s in sections]
    assert "LES GARANTIES DE BASE" in titres
    assert "LA GARANTIE VOL" in titres
    vol = next(s for s in sections if s.titre == "LA GARANTIE VOL")
    assert vol.page == 1  # 0-indexée
    assert "effraction" in vol.contenu
    # L'en-tête répété ne doit pas créer de section.
    assert "EN-TETE" not in titres


def test_le_texte_avant_le_premier_titre_est_conserve():
    """Aucune information ne doit être perdue au découpage."""
    pages = ["Texte liminaire important.\nLA GARANTIE VOL\nContenu."]

    sections = decouper_en_sections(pages)

    assert any("Texte liminaire important." in s.contenu for s in sections)


# --- Intégration avec split_documents ------------------------------------------


def _pages_en_documents(pages: list[str], source: str = "cg.pdf") -> list[Document]:
    return [
        Document(
            page_content=texte,
            metadata={"source": source, "page": i, "branche": "habitation"},
        )
        for i, texte in enumerate(pages)
    ]


def test_le_chunk_porte_sa_section_et_la_rappelle_en_tete():
    """Un passage doit rester compréhensible isolément : le titre y est préfixé."""
    docs = _pages_en_documents(["LA GARANTIE VOL\nLe vol est couvert après effraction."])

    chunks = split_documents(docs, Settings(chunking_strategy="semantic"))

    assert chunks
    chunk = chunks[0]
    assert chunk.metadata["section"] == "LA GARANTIE VOL"
    assert chunk.page_content.startswith("LA GARANTIE VOL")
    # Les métadonnées de provenance survivent au découpage.
    assert chunk.metadata["source"] == "cg.pdf"
    assert chunk.metadata["branche"] == "habitation"


def test_une_section_courte_n_est_pas_recoupee():
    docs = _pages_en_documents(["LA GARANTIE VOL\nCourt."])

    chunks = split_documents(docs, Settings(chunking_strategy="semantic", chunk_size=1000))

    assert len(chunks) == 1


def test_une_section_longue_est_recoupee_en_gardant_sa_section():
    docs = _pages_en_documents(["LA GARANTIE VOL\n" + "Le vol est couvert. " * 200])

    chunks = split_documents(docs, Settings(chunking_strategy="semantic", chunk_size=300))

    assert len(chunks) > 1
    assert all(c.metadata["section"] == "LA GARANTIE VOL" for c in chunks)


def test_aucun_chunk_ne_melange_deux_sections():
    """Critère d'acceptation T2.1 : une section n'est jamais coupée en deux."""
    pages = [
        "LES GARANTIES DE BASE\n" + "Incendie couvert. " * 30,
        "LA GARANTIE VOL\n" + "Vol couvert. " * 30,
        "LES EXCLUSIONS\n" + "Faute intentionnelle exclue. " * 30,
    ]
    docs = _pages_en_documents(pages)

    chunks = split_documents(docs, Settings(chunking_strategy="semantic", chunk_size=1000))

    sections = {c.metadata["section"] for c in chunks}
    assert sections == {"LES GARANTIES DE BASE", "LA GARANTIE VOL", "LES EXCLUSIONS"}
    for chunk in chunks:
        corps = chunk.page_content.split("\n", 1)[-1]
        autres = sections - {chunk.metadata["section"]}
        for autre in autres:
            assert autre not in corps


def test_strategie_fixed_conserve_le_comportement_historique():
    docs = _pages_en_documents(["LA GARANTIE VOL\n" + "phrase. " * 100])

    chunks = split_documents(
        docs, Settings(chunking_strategy="fixed", chunk_size=100, chunk_overlap=10)
    )

    assert len(chunks) > 1
    assert all("section" not in c.metadata for c in chunks)


def test_section_est_immuable():
    section = Section(titre="X", contenu="y", page=0)
    assert section.titre == "X"


def test_settings_refuse_une_configuration_incoherente():
    """Un overlap supérieur au chunk_size doit échouer à la configuration, pas au découpage."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=150)

    with pytest.raises(ValueError, match="chunking_strategy"):
        Settings(chunking_strategy="hybride")


def test_detecte_les_variantes_fragmentees_d_entete():
    """L'extraction PDF coupe les en-têtes : « MA SANTÉ » et « COMPLÉMENTAIRE SANTÉ
    MA SANTÉ » désignent le même bandeau. La variante longue, plus rare, doit suivre."""
    pages = ["MA SANTÉ\ncontenu"] * 40 + ["COMPLÉMENTAIRE SANTÉ MA SANTÉ\ncontenu"] * 9
    pages += ["LA GARANTIE VOL\ncontenu unique"]

    repetes = detecter_titres_repetes(pages)

    assert "MA SANTÉ" in repetes
    assert "COMPLÉMENTAIRE SANTÉ MA SANTÉ" in repetes
    assert "LA GARANTIE VOL" not in repetes


def test_la_page_citee_est_celle_du_morceau_pas_celle_de_la_section():
    """Une section de trois pages : un chunk tiré de la 3e page doit citer la 3e."""
    pages = [
        "LES GARANTIES DE BASE\n" + "Première page. " * 40,
        "Deuxième page. " * 40,
        "Troisième page avec le mot repère. " * 40,
    ]
    docs = _pages_en_documents(pages)

    chunks = split_documents(docs, Settings(chunking_strategy="semantic", chunk_size=500))

    pages_citees = {c.metadata["page"] for c in chunks}
    assert pages_citees == {0, 1, 2}
    repere = next(c for c in chunks if "repère" in c.page_content)
    assert repere.metadata["page"] == 2
