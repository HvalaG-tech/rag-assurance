"""Interface Streamlit — « Ask your insurance data ».

Une conversation avec un corpus de conditions générales d'assurance : chaque réponse
est sourcée passage par passage, chaque source affiche son score de pertinence et
la portion effectivement citée, et le système dit explicitement quand le corpus ne
permet pas de répondre.
"""

from __future__ import annotations

import html
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, Settings, settings
from app.demo import PARCOURS, charger_reponses_demo
from app.embeddings import get_vectorstore
from app.rag_chain import RagAnswer, answer

GITHUB_URL = "https://github.com/HvalaG-tech/rag-assurance"
EVAL_PATH = ROOT / "eval" / "results" / "latest.md"

COULEURS_CONFIANCE = {"élevée": "#1B7F3B", "moyenne": "#B7791F", "faible": "#B42318"}

st.set_page_config(
    page_title="Ask your insurance data",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    mark { background: #FFF1A8; padding: 0 2px; border-radius: 2px; }
    .badge { display:inline-block; padding:2px 10px; border-radius:12px; color:white;
             font-size:0.85rem; font-weight:600; }
    .extrait { font-size:0.92rem; line-height:1.45; white-space:pre-wrap; }
    .abstention { border-left: 5px solid #B42318; background:#FDF2F2; padding:12px 16px;
                  border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- État de session ------------------------------------------------------------


def _init_session() -> None:
    st.session_state.setdefault("historique", [])  # list[dict(question, reponse)]
    st.session_state.setdefault("question_en_attente", None)
    st.session_state.setdefault("cle_visiteur", "")
    st.session_state.setdefault("questions_posees", 0)


_init_session()


@st.cache_resource(show_spinner="Ouverture de l'index…")
def _vectorstore():
    return get_vectorstore(settings)


@st.cache_resource
def _reponses_demo():
    return charger_reponses_demo()


@st.cache_data
def _nombre_de_chunks() -> int:
    try:
        return len(_vectorstore().get(include=[])["ids"])
    except Exception:  # index absent : l'interface doit quand même s'afficher
        return 0


def _secret(nom: str) -> str | None:
    """Lit un réglage dans l'environnement, puis dans les secrets Streamlit.

    Sur Streamlit Community Cloud, les secrets sont normalement exposés comme
    variables d'environnement, mais `st.secrets` reste la source fiable — et sa
    simple lecture lève si aucun fichier de secrets n'existe (cas du poste local).
    """
    if valeur := os.getenv(nom):
        return valeur
    try:
        return st.secrets.get(nom)
    except Exception:
        return None


def _cle_serveur_disponible() -> bool:
    """Une clé API est-elle configurée côté serveur ?"""
    return bool(_secret("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API_KEY_BACKUP"))


# Sans clé côté serveur, aucun appel au modèle n'est possible : l'application est
# de facto en démonstration. Le déduire plutôt que de dépendre du seul réglage
# DEMO_MODE évite qu'un secret oublié au déploiement ne fasse planter la page sur
# une erreur d'authentification.
MODE_DEMO = (
    settings.demo_mode
    or (_secret("DEMO_MODE") or "").lower() in {"1", "true", "yes"}
    or not _cle_serveur_disponible()
)


# --- Panneau latéral ------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    n_chunks = _nombre_de_chunks()
    st.caption(f"{n_chunks} passages indexés · 4 contrats · 4 branches")

    mode = st.selectbox(
        "Recherche",
        ["hybrid", "vector", "bm25"],
        index=0,
        help="hybride = vectoriel + BM25 fusionnés (RRF). Les deux autres isolent un moteur.",
    )
    top_k = st.slider("Passages transmis au modèle", 3, 8, settings.top_k)
    reranker = st.toggle(
        "Reranker (cross-encoder)",
        value=settings.use_reranker,
        help="Réordonne les candidats par pertinence réelle. Coûteux en mémoire : "
        "désactivé sur la démo en ligne, chiffré en évaluation.",
    )

    st.divider()
    st.subheader("Votre clé API")
    st.session_state.cle_visiteur = st.text_input(
        "Clé Anthropic",
        value=st.session_state.cle_visiteur,
        type="password",
        help="Les questions libres utilisent votre clé. Elle reste dans votre session, "
        "n'est ni journalisée ni enregistrée.",
        label_visibility="collapsed",
        placeholder="sk-ant-… (questions libres uniquement)",
    )
    if MODE_DEMO:
        st.caption(
            "Mode démo : les six questions du parcours sont servies sans clé. "
            "Les questions libres utilisent la vôtre."
        )
    restantes = settings.max_questions_par_session - st.session_state.questions_posees
    st.caption(f"Questions libres restantes dans cette session : {max(restantes, 0)}")

    st.divider()
    with st.expander("Comment ce système est évalué"):
        if EVAL_PATH.exists():
            st.markdown(EVAL_PATH.read_text(encoding="utf-8"))
        else:
            st.markdown(
                "40 questions annotées à la main sur le corpus — 25 factuelles, 8 de synthèse, "
                "7 sans réponse dans le corpus. Métriques de recherche (recall@k, MRR) et de "
                "réponse (justesse, fidélité jugée par LLM, taux d'abstention correcte). "
                "Détail : `docs/EVALUATION.md`."
            )

    st.divider()
    if st.button("Effacer la conversation"):
        st.session_state.historique = []
        st.rerun()


def _configuration_courante() -> Settings:
    return replace(
        settings,
        retrieval_mode=mode,
        top_k=top_k,
        use_reranker=reranker,
        anthropic_api_key=st.session_state.cle_visiteur or None,
    )


# --- En-tête et parcours guidé --------------------------------------------------

st.title("🛡️ Ask your insurance data")
st.markdown(
    "Vos gestionnaires passent vingt minutes à chercher une règle dans trois cents pages de "
    "conditions générales. Posez la question ; obtenez la réponse, **le passage exact qui la "
    "fonde**, et un aveu clair quand le corpus ne la contient pas. "
    f"[Code source]({GITHUB_URL})"
)

st.markdown("##### Parcours de démonstration — à cliquer dans l'ordre")
colonnes = st.columns(3)
for i, etape in enumerate(PARCOURS):
    with colonnes[i % 3]:
        if st.button(
            f"{etape['etiquette']} — {etape['question']}",
            key=f"parcours_{i}",
            help=etape["pourquoi"],
            use_container_width=True,
        ):
            st.session_state.question_en_attente = etape["question"]


# --- Rendu d'une réponse --------------------------------------------------------


def _score_normalise(meta: dict, cfg: Settings) -> float:
    score = meta.get("score")
    if score is None:
        return 0.0
    if "score_rrf" in meta:  # logit du cross-encoder : -6 (hors sujet) -> 0 (exact)
        return max(0.0, min(1.0, (score + 6) / 6))
    return min(score / (2 / (cfg.rrf_k + 1)), 1.0)


_MOTS = re.compile(r"[\wÀ-ÿ]{4,}")


def _surligner(extrait: str, reponse: str) -> str:
    """Surligne dans l'extrait les phrases dont la réponse reprend le contenu.

    Une phrase est considérée citée quand la majorité de ses mots significatifs
    figurent dans la réponse — assez robuste aux reformulations pour rester lisible.
    """
    mots_reponse = {m.lower() for m in _MOTS.findall(reponse)}
    phrases = re.split(r"(?<=[\.\;\:])\s+|\n", extrait)
    rendu = []
    for phrase in phrases:
        if not phrase.strip():
            continue
        mots = [m.lower() for m in _MOTS.findall(phrase)]
        echappe = html.escape(phrase)
        if len(mots) >= 4 and sum(m in mots_reponse for m in mots) / len(mots) >= 0.6:
            rendu.append(f"<mark>{echappe}</mark>")
        else:
            rendu.append(echappe)
    return "\n".join(rendu)


def _badge(niveau: str) -> str:
    couleur = COULEURS_CONFIANCE.get(niveau, "#6B7280")
    return f'<span class="badge" style="background:{couleur}">confiance {niveau}</span>'


def _afficher_reponse(reponse: RagAnswer, cfg: Settings) -> None:
    if reponse.abstention:
        st.markdown(
            '<div class="abstention"><strong>Information absente du corpus.</strong><br>'
            "Le système a cherché, n'a pas trouvé de passage qui réponde, et le dit — "
            "plutôt que de produire une réponse plausible mais infondée.</div>",
            unsafe_allow_html=True,
        )
        corps = re.sub(r"^[\s\*_#>`-]*INFORMATION ABSENTE DU CORPUS\.?\**", "", reponse.answer)
        if corps.strip():
            st.markdown(corps.strip())
    else:
        st.markdown(reponse.answer)
        detail = f"recherche {reponse.confiance_recherche} · modèle {reponse.confiance_modele}"
        st.markdown(
            _badge(reponse.confiance)
            + f" <span style='color:#6B7280;font-size:0.85rem'>{detail}</span>",
            unsafe_allow_html=True,
        )

    if reponse.usage:
        st.caption(
            f"{reponse.usage.get('input_tokens', 0)} tokens en entrée · "
            f"{reponse.usage.get('output_tokens', 0)} en sortie · "
            f"≈ {reponse.cout_estime:.3f} $"
        )

    if not reponse.sources:
        return

    st.markdown(
        "**Passages sources**" + (" (aucun n'a été jugé suffisant)" if reponse.abstention else "")
    )
    for i, doc in enumerate(reponse.sources, start=1):
        meta = doc.metadata
        page = meta.get("page")
        ref = f"p.{page + 1}" if isinstance(page, int) else ""
        score = _score_normalise(meta, cfg)
        rangs = meta.get("rangs") or {}
        detail_rangs = " · ".join(f"{k} #{v}" for k, v in rangs.items())
        titre = (
            f"{i}. {meta.get('assureur', '?').upper()} · {meta.get('branche', '?')} · "
            f"{meta.get('source', '?')} {ref} — {meta.get('section', '')[:60]}"
        )
        with st.expander(titre, expanded=(i == 1 and not reponse.abstention)):
            st.progress(
                score,
                text=f"pertinence {score:.0%}" + (f" — {detail_rangs}" if detail_rangs else ""),
            )
            st.markdown(
                f'<div class="extrait">{_surligner(doc.page_content, reponse.answer)}</div>',
                unsafe_allow_html=True,
            )


# --- Historique -----------------------------------------------------------------

for tour in st.session_state.historique:
    with st.chat_message("user"):
        st.write(tour["question"])
    with st.chat_message("assistant"):
        _afficher_reponse(tour["reponse"], tour["cfg"])


# --- Nouvelle question -----------------------------------------------------------


def _repondre(question: str) -> tuple[RagAnswer, Settings] | None:
    cfg = _configuration_courante()

    if MODE_DEMO and question in _reponses_demo():
        return _reponses_demo()[question], replace(cfg, use_reranker=True)

    # Sans clé du visiteur, on refuse dans deux cas : en mode démo, où les questions
    # libres ne doivent jamais être facturées au propriétaire de la démo ; et quand
    # aucune clé n'existe nulle part — sans quoi le SDK lève une erreur
    # d'authentification que Streamlit affiche en trace Python.
    if not cfg.anthropic_api_key and (MODE_DEMO or not _cle_serveur_disponible()):
        st.warning(
            "Cette question sort du parcours de démonstration : elle demande un appel "
            "au modèle, donc **votre propre clé API** — à saisir dans le panneau latéral "
            "(elle reste dans votre session, n'est ni journalisée ni enregistrée).\n\n"
            "Les six questions du parcours ci-dessus fonctionnent sans clé."
        )
        return None

    if st.session_state.questions_posees >= settings.max_questions_par_session:
        st.warning("Plafond de questions atteint pour cette session.")
        return None

    st.session_state.questions_posees += 1
    with st.spinner("Recherche dans les contrats, puis rédaction…"):
        return answer(question, cfg), cfg


question = st.chat_input("Posez votre question sur les contrats indexés…")
if st.session_state.question_en_attente:
    question = st.session_state.question_en_attente
    st.session_state.question_en_attente = None

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        resultat = _repondre(question)
        if resultat:
            reponse, cfg = resultat
            _afficher_reponse(reponse, cfg)
            st.session_state.historique.append(
                {"question": question, "reponse": reponse, "cfg": cfg}
            )
