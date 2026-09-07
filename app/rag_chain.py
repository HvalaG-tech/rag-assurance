"""Pipeline RAG : recherche hybride + génération sourcée, avec abstention et confiance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.config import (
    MARQUEUR_ABSTENTION,
    PRIX_PAR_MILLION,
    SYSTEM_PROMPT,
    USER_PROMPT,
    Settings,
    settings,
)
from app.embeddings import get_vectorstore
from app.retrieval import rechercher

NIVEAUX = ("faible", "moyenne", "élevée")

# Seuils sur le logit du cross-encoder mmarco (voir `confiance_recherche`).
SEUILS_RERANKER = {"élevée": -1.0, "moyenne": -3.5}

_MOTIF_CONFIANCE = re.compile(r"\n?\s*CONFIANCE\s*:\s*(élevée|elevee|moyenne|faible)\.?\s*$", re.I)


@dataclass
class RagAnswer:
    """Réponse du système, avec tout ce qu'il faut pour l'auditer."""

    answer: str
    sources: list[Document]
    abstention: bool = False
    # Indice global (élevée / moyenne / faible), ou "abstention".
    confiance: str = "moyenne"
    # Auto-évaluation brute du modèle et signal de recherche, pour expliquer l'indice.
    confiance_modele: str = "moyenne"
    confiance_recherche: str = "moyenne"
    usage: dict[str, int] = field(default_factory=dict)
    cout_estime: float = 0.0


def get_llm(cfg: Settings = settings) -> BaseChatModel:
    """Instancie le LLM selon le provider configuré."""
    if cfg.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.openai_model, max_tokens=cfg.max_tokens)

    from langchain_anthropic import ChatAnthropic

    parametres: dict = {
        "model": cfg.anthropic_model,
        "max_tokens": cfg.max_tokens,
        # `effort` arbitre profondeur de raisonnement vs coût/latence.
        # Les modèles Claude 5 n'acceptent plus temperature / top_p / budget_tokens.
        "output_config": {"effort": cfg.effort},
    }
    if cfg.anthropic_api_key:
        parametres["api_key"] = cfg.anthropic_api_key
    return ChatAnthropic(**parametres)


def format_context(docs: list[Document]) -> str:
    """Met en forme les extraits avec leur provenance, pour permettre la citation.

    L'en-tête porte l'assureur, la branche et la section en plus du fichier : le
    modèle sait ainsi *de quel contrat* vient chaque passage. Sans cela, il lit
    « AllSecur » dans un extrait, « Allianz » dans la question, et s'abstient à
    tort — AllSecur étant la marque directe d'Allianz, l'écart n'est que de nom.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = meta.get("source", "inconnu")
        page = meta.get("page")
        ref = f"{source}, p.{page + 1}" if isinstance(page, int) else source
        identite = []
        if meta.get("assureur") and meta["assureur"] != "inconnu":
            identite.append(f"assureur : {meta['assureur'].upper()}")
        if meta.get("branche") and meta["branche"] != "inconnu":
            identite.append(f"branche : {meta['branche']}")
        if meta.get("section"):
            identite.append(f"section : {meta['section']}")
        entete = f"[Extrait {i} — Source: {ref}" + (
            f" — {' · '.join(identite)}]" if identite else "]"
        )
        blocks.append(f"{entete}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def retrieve(question: str, cfg: Settings = settings, filtre: dict | None = None) -> list[Document]:
    """Sélectionne les passages pertinents selon le mode de recherche configuré."""
    return rechercher(question, get_vectorstore(cfg), cfg, filtre=filtre)


def confiance_recherche(sources: list[Document], cfg: Settings = settings) -> str:
    """Traduit le score du meilleur passage en niveau lisible.

    Score RRF : borné par `2/(k+1)` (1er des deux moteurs) — on le normalise.
    Score cross-encoder : un logit. Sur ce modèle (mmarco), un passage qui répond
    exactement se situe vers 0 (-0,2 observé sur « délai de déclaration du vol »),
    un passage voisin vers -3, un passage hors sujet sous -4. Les seuils viennent
    de ces observations sur le corpus, pas d'une sigmoïde théorique.
    """
    if not sources:
        return "faible"
    meta = sources[0].metadata
    score = meta.get("score")
    if score is None:
        return "moyenne"
    if "score_rrf" in meta:  # le reranker a tourné : `score` est son logit
        if score >= SEUILS_RERANKER["élevée"]:
            return "élevée"
        if score >= SEUILS_RERANKER["moyenne"]:
            return "moyenne"
        return "faible"
    normalise = score / (2 / (cfg.rrf_k + 1))
    if normalise >= 0.6:
        return "élevée"
    if normalise >= 0.3:
        return "moyenne"
    return "faible"


def combiner_confiance(recherche: str, modele: str) -> str:
    """Indice global : le plus prudent des deux signaux l'emporte.

    « élevée » exige que ni la recherche ni le modèle ne doutent. Un seul signal
    « faible » suffit à afficher « faible » — en assurance, mieux vaut sous-promettre.
    """
    rang = min(NIVEAUX.index(recherche), NIVEAUX.index(modele))
    return NIVEAUX[rang]


# Toute chaîne ressemblant à une clé, pour ne jamais la réafficher : les messages
# d'erreur des fournisseurs en citent parfois un fragment.
_MOTIF_CLE = re.compile(r"\b(sk|pk)-[A-Za-z0-9_\-]{6,}", re.IGNORECASE)

# Signature textuelle -> message destiné au visiteur. L'ordre compte : « crédit
# insuffisant » arrive dans une erreur 400 qu'il ne faut pas confondre avec une
# clé invalide.
_DIAGNOSTICS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("credit balance", "billing", "insufficient_quota", "quota"),
        "Le crédit associé à cette clé API est épuisé. Rechargez le compte chez votre "
        "fournisseur, ou essayez une autre clé.",
    ),
    (
        ("authentication_error", "invalid x-api-key", "401", "invalid_api_key", "unauthorized"),
        "Cette clé API a été refusée : elle est invalide, révoquée, ou incomplète. "
        "Vérifiez-la dans le panneau latéral — elle doit commencer par « sk-ant- » et "
        "être copiée en entier.",
    ),
    (
        ("permission", "403", "forbidden"),
        "Cette clé API n'a pas les droits nécessaires pour interroger le modèle. "
        "Vérifiez ses permissions chez votre fournisseur.",
    ),
    (
        ("rate_limit", "429", "too many requests"),
        "Trop de requêtes envoyées avec cette clé en peu de temps. Patientez une minute "
        "avant de réessayer.",
    ),
    (
        ("connection", "timeout", "timed out", "network", "dns"),
        "Impossible de joindre le service du modèle : le réseau ne répond pas. "
        "Réessayez dans un instant.",
    ),
    (
        ("overloaded", "529", "503", "502", "500", "internal server"),
        "Le service du modèle est momentanément indisponible ou surchargé. "
        "Réessayez dans un instant.",
    ),
)

MESSAGE_ERREUR_GENERIQUE = (
    "La réponse n'a pas pu être générée. Réessayez ; si le problème persiste, "
    "vérifiez la clé API saisie dans le panneau latéral."
)


def message_erreur_api(erreur: BaseException) -> str:
    """Traduit une erreur d'appel au modèle en message compréhensible.

    On s'appuie sur le texte plutôt que sur les classes d'exception du SDK : LangChain
    enveloppe et ré-emballe les erreurs, et l'application doit rester lisible quel que
    soit le fournisseur. Le message brut n'est jamais réaffiché — il peut contenir un
    fragment de la clé.
    """
    texte = f"{type(erreur).__name__} {erreur}".lower()
    for signatures, message in _DIAGNOSTICS:
        if any(signature in texte for signature in signatures):
            return message
    return MESSAGE_ERREUR_GENERIQUE


def masquer_les_cles(texte: str) -> str:
    """Remplace toute chaîne ressemblant à une clé API par un marqueur."""
    return _MOTIF_CLE.sub("[clé masquée]", texte)


def texte_de_la_reponse(contenu: str | list) -> str:
    """Ne garde que le texte d'une réponse de modèle.

    Avec le raisonnement adaptatif, Claude renvoie une liste de blocs (`thinking`,
    `text`, …) et non une chaîne : seul le texte destiné à l'utilisateur compte.
    """
    if isinstance(contenu, str):
        return contenu
    morceaux = []
    for bloc in contenu:
        if isinstance(bloc, str):
            morceaux.append(bloc)
        elif isinstance(bloc, dict) and bloc.get("type") == "text":
            morceaux.append(bloc.get("text", ""))
    return "".join(morceaux)


def est_une_abstention(corps: str) -> bool:
    """Vrai si la réponse s'ouvre sur le marqueur d'abstention.

    Le modèle le met volontiers en gras ou en titre Markdown : on ignore la
    ponctuation de mise en forme avant de comparer.
    """
    debut = corps.lstrip(" \n\t*_#>`-").upper()
    return debut.startswith(MARQUEUR_ABSTENTION.rstrip("."))


def _extraire_confiance(texte: str) -> tuple[str, str]:
    """Sépare la ligne `CONFIANCE: …` du corps de la réponse."""
    match = _MOTIF_CONFIANCE.search(texte)
    if not match:
        return texte.strip(), "moyenne"
    niveau = match.group(1).lower().replace("elevee", "élevée")
    return texte[: match.start()].strip(), niveau


def _cout(usage: dict[str, int], cfg: Settings) -> float:
    entree, sortie = PRIX_PAR_MILLION.get(cfg.anthropic_model, (0.0, 0.0))
    return (usage.get("input_tokens", 0) * entree + usage.get("output_tokens", 0) * sortie) / 1e6


def answer(
    question: str,
    cfg: Settings = settings,
    filtre: dict | None = None,
    sources: list[Document] | None = None,
) -> RagAnswer:
    """Répond à une question en langage naturel, sources à l'appui.

    `sources` permet de réutiliser une recherche déjà faite (évaluation, interface).
    """
    docs = sources if sources is not None else retrieve(question, cfg, filtre)
    if not docs:
        return RagAnswer(
            answer=f"{MARQUEUR_ABSTENTION} Aucun document indexé ne permet de répondre.",
            sources=[],
            abstention=True,
            confiance="abstention",
            confiance_recherche="faible",
        )

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])
    chain = prompt | get_llm(cfg)
    response = chain.invoke({"context": format_context(docs), "question": question})

    texte = texte_de_la_reponse(response.content)
    corps, niveau_modele = _extraire_confiance(texte)
    abstention = est_une_abstention(corps)

    usage = {}
    if getattr(response, "usage_metadata", None):
        usage = {
            "input_tokens": int(response.usage_metadata.get("input_tokens", 0)),
            "output_tokens": int(response.usage_metadata.get("output_tokens", 0)),
        }

    niveau_recherche = confiance_recherche(docs, cfg)
    return RagAnswer(
        answer=corps,
        sources=docs,
        abstention=abstention,
        confiance="abstention"
        if abstention
        else combiner_confiance(niveau_recherche, niveau_modele),
        confiance_modele=niveau_modele,
        confiance_recherche=niveau_recherche,
        usage=usage,
        cout_estime=_cout(usage, cfg),
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Quelles sont les exclusions de garantie incendie ?"
    result = answer(q)
    print(result.answer)
    print(f"\n[confiance : {result.confiance} · coût ≈ {result.cout_estime:.4f} $]")
