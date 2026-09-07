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


def test_mode_demo_repond_sans_cle_api(monkeypatch):
    """Comportement critique de la démo publique : un visiteur sans clé clique sur
    une question du parcours et obtient une vraie réponse sourcée, sans appel API."""
    from app.config import settings
    from app.rag_chain import answer as _answer

    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    # Tout appel au LLM pendant ce test est un échec : la réponse doit être pré-calculée.
    monkeypatch.setattr(
        "app.main.answer",
        lambda *a, **k: pytest.fail("le mode démo ne doit appeler aucune API"),
        raising=False,
    )

    test = AppTest.from_file(str(MAIN), default_timeout=120)
    test.run()
    test.button(key="parcours_0").click().run()

    assert not test.exception, test.exception
    # La réponse pré-calculée cite bien un passage du contrat.
    rendu = " ".join(m.value for m in test.markdown)
    assert "2 jours ouvrés" in rendu
    assert "dg_allsecur_auto.pdf" in rendu
    assert _answer  # l'import réel existe : le monkeypatch a bien remplacé un symbole présent


def test_mode_demo_refuse_une_question_libre_sans_cle(monkeypatch):
    """Garde-fou budgétaire : hors parcours, il faut la clé du visiteur."""
    from app.config import settings

    monkeypatch.setattr(settings, "demo_mode", True)

    test = AppTest.from_file(str(MAIN), default_timeout=120)
    test.run()
    test.chat_input[0].set_value("Quelle est la franchise en cas de grêle ?").run()

    assert not test.exception, test.exception
    assert any("votre propre clé" in w.value for w in test.warning), [w.value for w in test.warning]


def _sans_aucune_cle(monkeypatch):
    """Reproduit l'environnement de Streamlit Cloud sans secret : aucune clé nulle part."""
    for nom in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_BACKUP", "OPENAI_API_KEY", "DEMO_MODE"):
        monkeypatch.delenv(nom, raising=False)
    from app.config import settings

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "anthropic_api_key", None)


def test_sans_cle_ni_reglage_demo_le_parcours_fonctionne_quand_meme(monkeypatch):
    """Régression du crash de mise en ligne : le secret DEMO_MODE avait été omis, et
    l'application appelait le modèle sans clé — trace d'authentification à l'écran.

    Sans clé serveur, aucun appel n'est possible : le mode démo doit s'imposer seul."""
    _sans_aucune_cle(monkeypatch)

    test = AppTest.from_file(str(MAIN), default_timeout=120)
    test.run()
    test.button(key="parcours_0").click().run()

    assert not test.exception, test.exception
    assert "2 jours ouvrés" in " ".join(m.value for m in test.markdown)


def test_sans_cle_une_question_libre_est_refusee_proprement(monkeypatch):
    """Et une question hors parcours affiche un message clair, jamais une trace Python."""
    _sans_aucune_cle(monkeypatch)

    test = AppTest.from_file(str(MAIN), default_timeout=120)
    test.run()
    test.chat_input[0].set_value("Quelle est la franchise en cas de grêle ?").run()

    assert not test.exception, test.exception
    assert any("votre propre clé" in w.value for w in test.warning), [w.value for w in test.warning]
