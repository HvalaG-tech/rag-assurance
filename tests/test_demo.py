"""Tests du mode démo : parcours guidé et aller-retour JSON des réponses."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.demo import (
    PARCOURS,
    charger_reponses_demo,
    deserialiser,
    sauvegarder_reponses_demo,
    serialiser,
)
from app.rag_chain import RagAnswer


def test_le_parcours_raconte_l_histoire_prevue():
    """Six étapes, dans l'ordre de la roadmap : l'abstention est en 4e position."""
    assert len(PARCOURS) == 6
    assert "Hors corpus" in PARCOURS[3]["etiquette"]
    assert len({e["question"] for e in PARCOURS}) == 6


def test_aller_retour_json_d_une_reponse(tmp_path: Path):
    reponse = RagAnswer(
        answer="Le délai est de 2 jours ouvrés.",
        sources=[
            Document(
                page_content="LA GARANTIE VOL\n2 jours ouvrés",
                metadata={"source": "a.pdf", "page": 3, "score": 0.03, "rangs": {"bm25": 1}},
            )
        ],
        confiance="élevée",
        usage={"input_tokens": 10, "output_tokens": 5},
        cout_estime=0.001,
    )
    chemin = tmp_path / "demo.json"

    sauvegarder_reponses_demo({"q ?": reponse}, chemin)
    relu = charger_reponses_demo(chemin)

    assert relu["q ?"].answer == reponse.answer
    assert relu["q ?"].sources[0].metadata["page"] == 3
    assert relu["q ?"].confiance == "élevée"
    assert deserialiser(serialiser(reponse)).cout_estime == 0.001


def test_fichier_absent_donne_un_dictionnaire_vide(tmp_path: Path):
    assert charger_reponses_demo(tmp_path / "absent.json") == {}
