"""Traduction des erreurs d'appel au modèle en messages lisibles (sans fuite de clé)."""

from __future__ import annotations

import pytest

from app.rag_chain import message_erreur_api


class FauxAuthenticationError(Exception):
    """Imite `anthropic.AuthenticationError` sans dépendre du SDK dans le test."""


class FauxRateLimitError(Exception):
    pass


class FauxAPIConnectionError(Exception):
    pass


def test_cle_invalide_donne_un_message_actionnable():
    erreur = FauxAuthenticationError(
        "Error code: 401 - {'error': {'type': 'authentication_error', "
        "'message': 'invalid x-api-key'}}"
    )

    message = message_erreur_api(erreur)

    assert "clé" in message.lower()
    assert "invalide" in message.lower() or "refusée" in message.lower()
    # Le visiteur doit savoir quoi faire.
    assert "panneau latéral" in message.lower() or "saisir" in message.lower()


def test_credit_epuise_est_distingue_d_une_cle_invalide():
    erreur = Exception(
        "Error code: 400 - {'error': {'type': 'invalid_request_error', 'message': "
        "'Your credit balance is too low to access the Anthropic API'}}"
    )

    message = message_erreur_api(erreur)

    assert "crédit" in message.lower()
    assert "invalide" not in message.lower()


def test_limite_de_requetes():
    message = message_erreur_api(FauxRateLimitError("Error code: 429 - rate_limit_error"))

    assert "trop de requêtes" in message.lower() or "limite" in message.lower()


def test_probleme_de_reseau():
    message = message_erreur_api(FauxAPIConnectionError("Connection error."))

    assert "joindre" in message.lower() or "réseau" in message.lower()


def test_erreur_inconnue_reste_comprehensible():
    message = message_erreur_api(Exception("Error code: 529 - overloaded_error"))

    assert message
    assert "réessayer" in message.lower() or "indisponible" in message.lower()


@pytest.mark.parametrize(
    ("brut", "partie_secrete"),
    [
        ("invalid x-api-key: sk-ant-api03-AbCdEf0123456789XyZ", "AbCdEf0123456789XyZ"),
        ("Bearer sk-ant-admin01-SECRETSECRETSECRET", "SECRETSECRETSECRET"),
        ("clé fournie : sk-proj-0123456789abcdefghij", "0123456789abcdefghij"),
    ],
)
def test_aucune_cle_n_est_jamais_reaffichee(brut: str, partie_secrete: str):
    """Un message d'erreur peut contenir la clé : sa partie secrète ne doit jamais
    ressortir. Le préfixe public « sk-ant- » figure en revanche dans l'aide à la saisie."""
    message = message_erreur_api(Exception(brut))

    assert partie_secrete not in message
    assert brut not in message


@pytest.mark.parametrize(
    ("brut", "partie_secrete"),
    [
        ("invalid x-api-key: sk-ant-api03-AbCdEf0123456789XyZ", "AbCdEf0123456789XyZ"),
        ("Bearer sk-ant-admin01-SECRETSECRETSECRET", "SECRETSECRETSECRET"),
        ("clé fournie : sk-proj-0123456789abcdefghij", "0123456789abcdefghij"),
    ],
)
def test_masquage_pour_les_journaux(brut: str, partie_secrete: str):
    """`masquer_les_cles` sert partout où un texte brut pourrait être journalisé."""
    from app.rag_chain import masquer_les_cles

    masque = masquer_les_cles(brut)

    assert partie_secrete not in masque
    assert "[clé masquée]" in masque


def test_le_message_reste_court():
    """Un pavé technique dans l'interface dessert le propos."""
    for erreur in [
        FauxAuthenticationError("401 invalid x-api-key"),
        Exception("credit balance is too low"),
        Exception("boom"),
    ]:
        assert len(message_erreur_api(erreur)) < 300
