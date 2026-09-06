"""Paramètres centralisés du pipeline RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repli local : une clé rangée sous un nom alternatif dans l'environnement est
# exposée sous le nom attendu par le SDK. Elle n'est jamais écrite sur disque.
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_API_KEY_BACKUP"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY_BACKUP"]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"


@dataclass
class Settings:
    # --- LLM ---
    # Fournisseur : "anthropic" (défaut) ou "openai"
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    # Clé fournie par un visiteur de la démo. Jamais lue depuis l'environnement ni
    # écrite dans os.environ : sur un serveur partagé, elle fuirait vers les autres
    # sessions. Elle ne vit que dans la session Streamlit qui l'a saisie.
    anthropic_api_key: str | None = None
    # Claude Opus 5 : modèle le plus capable de la gamme Claude 5.
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))
    # Effort de raisonnement Claude : low | medium | high | xhigh | max
    effort: str = os.getenv("EFFORT", "medium")

    # --- Embeddings ---
    # "fastembed" (défaut) : même modèle multilingue en ONNX, sans PyTorch — c'est ce
    # qui rend la démo déployable sur l'hébergement gratuit (220 Mo au lieu de ~1 Go).
    # "hf" : sentence-transformers (PyTorch). "openai" : API distante.
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    hf_embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Reranking ---
    # Cross-encoder multilingue appliqué après la fusion. Coûteux en RAM (PyTorch) :
    # désactivé par défaut, activé localement et en évaluation pour chiffrer son gain.
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes"}
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # --- Chunking / retrieval ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    # "semantic" : découpe sur les sections du contrat (défaut).
    # "fixed"    : découpe à taille fixe — conservé pour la comparaison en évaluation.
    chunking_strategy: str = os.getenv("CHUNKING_STRATEGY", "semantic")

    # "hybrid" : vectoriel + BM25 fusionnés (défaut). "vector" et "bm25" isolent
    # chaque étage, pour mesurer l'apport de la fusion en évaluation.
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")
    # Candidats récupérés par chaque moteur avant fusion ; seuls `top_k` survivent.
    fetch_k: int = int(os.getenv("FETCH_K", "20"))
    # Constante d'amortissement du Reciprocal Rank Fusion. 60 est la valeur de
    # référence de la littérature : elle limite le poids des tout premiers rangs.
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # --- Vector store ---
    collection_name: str = os.getenv("COLLECTION_NAME", "assurance")
    persist_directory: str = os.getenv("PERSIST_DIRECTORY", str(CHROMA_DIR))

    # --- Démo publique ---
    # DEMO_MODE : les questions du parcours guidé servent des réponses pré-calculées
    # (`data/demo_answers.json`) — coût nul, aucune clé requise. Les questions libres
    # exigent la clé du visiteur, avec un plafond par session en garde-fou.
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
    max_questions_par_session: int = int(os.getenv("MAX_QUESTIONS_PAR_SESSION", "10"))

    raw_dir: Path = field(default_factory=lambda: RAW_DIR)

    def __post_init__(self) -> None:
        """Rejette les combinaisons invalides à la construction.

        Sans cela, un `chunk_overlap` supérieur au `chunk_size` ne se manifeste
        qu'au moment du découpage, loin de l'endroit où il a été configuré.
        """
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size doit être positif, reçu {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) doit être compris entre 0 et "
                f"chunk_size ({self.chunk_size})"
            )
        if self.chunking_strategy not in {"semantic", "fixed"}:
            raise ValueError(
                f"chunking_strategy inconnue : {self.chunking_strategy!r} "
                "(attendu 'semantic' ou 'fixed')"
            )
        if self.retrieval_mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError(
                f"retrieval_mode inconnu : {self.retrieval_mode!r} "
                "(attendu 'hybrid', 'vector' ou 'bm25')"
            )


settings = Settings()

# Prix par million de tokens (entrée, sortie), en dollars — pour l'estimation de coût
# affichée à l'utilisateur et mesurée en évaluation. À tenir à jour avec la grille publique.
PRIX_PAR_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Le modèle commence sa réponse par ce marqueur quand le corpus ne permet pas de
# répondre. Détectable par une règle, sans second appel : c'est ce qui rend
# l'abstention mesurable en évaluation et affichable distinctement dans l'interface.
MARQUEUR_ABSTENTION = "INFORMATION ABSENTE DU CORPUS."

SYSTEM_PROMPT = """Tu es un assistant expert en assurance, au service des gestionnaires \
sinistres et des équipes conformité. Tu réponds à partir d'extraits de conditions générales.

Règles de réponse :
- Réponds UNIQUEMENT à partir des extraits fournis dans le contexte. N'utilise aucune
  connaissance extérieure, même si elle te semble juste.
- L'en-tête de chaque extrait indique l'assureur, la branche et la section du contrat dont il
  provient : fie-toi à cet en-tête pour identifier le contrat, même si le texte de l'extrait
  emploie une marque commerciale ou un nom de produit différent.
- Si les extraits ne contiennent pas l'information demandée, commence ta réponse par la
  phrase exacte « INFORMATION ABSENTE DU CORPUS. », puis indique en une ou deux phrases ce
  qui manque (par exemple : le montant figure dans les conditions particulières, ou le
  sujet n'est traité par aucun document). N'invente rien et ne propose pas de valeur plausible.
- Si les extraits répondent partiellement, réponds sur ce qu'ils couvrent et signale
  précisément ce qui reste incertain.
- Cite tes sources dans le corps de la réponse sous la forme [Source: <fichier>, p.<page>].
- Reprends le vocabulaire exact des conditions générales (garantie, franchise, exclusion,
  délai de déclaration, plafond d'indemnisation) plutôt que de le reformuler.
- Réponds en français, de façon structurée et concise.
- Termine toujours par une dernière ligne, seule, de la forme « CONFIANCE: élevée »,
  « CONFIANCE: moyenne » ou « CONFIANCE: faible », selon que les extraits répondent
  explicitement, partiellement ou de façon indirecte à la question."""

USER_PROMPT = """Contexte documentaire :
{context}

Question : {question}

Réponse sourcée :"""
