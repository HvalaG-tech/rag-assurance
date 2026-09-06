"""L'application Streamlit s'exécute sans erreur, en mode headless.

`AppTest` rejoue le script complet : import des modules, ouverture de l'index,
rendu du parcours guidé. Aucun appel LLM n'est déclenché tant qu'aucune question
n'est posée — le test reste gratuit et hors ligne.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.demo import PARCOURS

MAIN = Path(__file__).resolve().parent.parent / "app" / "main.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    test = AppTest.from_file(str(MAIN), default_timeout=120)
    test.run()
    return test


def test_l_application_demarre_sans_exception(app: AppTest):
    assert not app.exception, app.exception


def test_le_parcours_guide_est_affiche_en_boutons(app: AppTest):
    libelles = [b.label for b in app.button]
    for etape in PARCOURS:
        assert any(etape["question"] in libelle for libelle in libelles), etape["question"]


def test_le_titre_et_la_promesse_sont_visibles(app: AppTest):
    assert any("Ask your insurance data" in t.value for t in app.title)
    assert any("passage exact" in m.value for m in app.markdown)


def test_la_cle_visiteur_est_un_champ_masque(app: AppTest):
    champs = [t for t in app.sidebar.text_input]
    assert champs, "le champ de clé API doit exister dans le panneau latéral"
