"""Tests des métadonnées documentaires déduites de l'arborescence (T1.2)."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.embeddings import CLES_METADONNEES, extraire_metadonnees, load_documents, split_documents


def test_extrait_branche_et_assureur_de_l_arborescence(tmp_path: Path):
    chemin = tmp_path / "habitation" / "mma" / "cg651o_habitation_2016-01.pdf"

    meta = extraire_metadonnees(chemin, tmp_path)

    assert meta["branche"] == "habitation"
    assert meta["assureur"] == "mma"


def test_deduit_le_type_de_document_du_prefixe():
    racine = Path("/corpus")
    cas = {
        "cg651o_habitation_2016-01.pdf": "CG",
        "dg_allsecur_auto.pdf": "DG",
        "ipid_collection608.pdf": "DIPA",
        "notice_itineo.pdf": "notice",
        "faq_sinistres.md": "FAQ",
    }
    for nom, attendu in cas.items():
        meta = extraire_metadonnees(racine / "auto" / "axa" / nom, racine)
        assert meta["type_doc"] == attendu, nom


def test_extrait_la_date_du_nom_de_fichier():
    racine = Path("/corpus")

    avec_mois = extraire_metadonnees(racine / "sante" / "axa" / "cg_x_2020-12.pdf", racine)
    assert avec_mois["date_doc"] == "2020-12"

    annee_seule = extraire_metadonnees(racine / "sante" / "axa" / "cg_x_2023.pdf", racine)
    assert annee_seule["date_doc"] == "2023"


def test_repli_sur_inconnu_hors_arborescence(tmp_path: Path):
    """Un fichier posé à la racine n'a ni branche ni assureur : rien ne doit casser."""
    meta = extraire_metadonnees(tmp_path / "document_libre.pdf", tmp_path)

    assert meta["branche"] == "inconnu"
    assert meta["assureur"] == "inconnu"
    assert meta["type_doc"] == "inconnu"
    assert meta["date_doc"] == "inconnu"


def test_les_quatre_cles_survivent_au_chargement_et_au_decoupage(tmp_path: Path):
    """Critère d'acceptation T1.2 : les quatre clés sont présentes sur CHAQUE chunk."""
    fichier = tmp_path / "auto" / "allianz" / "dg_allsecur_auto_2024-03.md"
    fichier.parent.mkdir(parents=True)
    fichier.write_text("Garantie responsabilité civile. " * 200, encoding="utf-8")

    docs = load_documents(tmp_path)
    chunks = split_documents(docs, Settings(chunk_size=100, chunk_overlap=10))

    assert len(chunks) > 1
    for chunk in chunks:
        for cle in CLES_METADONNEES:
            assert chunk.metadata.get(cle), f"{cle} manquante sur un chunk"
        assert chunk.metadata["branche"] == "auto"
        assert chunk.metadata["assureur"] == "allianz"
        assert chunk.metadata["type_doc"] == "DG"
        assert chunk.metadata["date_doc"] == "2024-03"


def test_chroma_accepte_les_metadonnees(tmp_path: Path):
    """Chroma refuse les valeurs non scalaires : tout doit rester en chaînes."""
    meta = extraire_metadonnees(tmp_path / "rc" / "smacl" / "cg_rc_2023-02.pdf", tmp_path)

    assert set(meta) == set(CLES_METADONNEES)
    assert all(isinstance(v, str) for v in meta.values())
