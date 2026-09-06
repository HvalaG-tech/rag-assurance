"""Ingestion documentaire : chargement, chunking, embeddings, vector store ChromaDB."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking import decouper_en_sections
from app.config import Settings, settings

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# Métadonnées de provenance portées par chaque chunk, en plus de `source` et `page`.
# Elles permettent le filtrage par branche ou assureur et une citation plus précise.
CLES_METADONNEES = ("branche", "assureur", "type_doc", "date_doc")

INCONNU = "inconnu"

# Préfixe du nom de fichier -> type de document normalisé.
_TYPES_DOC = {
    "cg": "CG",
    "dg": "DG",
    "ipid": "DIPA",
    "dipa": "DIPA",
    "notice": "notice",
    "faq": "FAQ",
}

# `cg651o_...` doit être reconnu comme CG : on ignore le suffixe de référence du contrat.
_MOTIF_TYPE = re.compile(r"^([a-z]+)", re.IGNORECASE)
_MOTIF_DATE = re.compile(r"(20\d{2})(?:-(\d{2}))?")


def extraire_metadonnees(chemin: Path, racine: Path) -> dict[str, str]:
    """Déduit branche, assureur, type de document et date depuis le chemin du fichier.

    Convention attendue : `<racine>/<branche>/<assureur>/<type><ref>_<sujet>_<AAAA-MM>.pdf`.
    Tout élément absent vaut `"inconnu"` — un fichier hors convention doit pouvoir être
    indexé sans faire échouer l'ingestion.
    """
    meta = dict.fromkeys(CLES_METADONNEES, INCONNU)

    try:
        parties = chemin.resolve().relative_to(racine.resolve()).parts
    except ValueError:
        parties = (chemin.name,)

    # parties = (branche, assureur, ..., nom_de_fichier)
    if len(parties) >= 3:
        meta["branche"] = parties[0].lower()
        meta["assureur"] = parties[1].lower()

    nom = chemin.stem.lower()

    if (prefixe := _MOTIF_TYPE.match(nom)) and (type_doc := _TYPES_DOC.get(prefixe.group(1))):
        meta["type_doc"] = type_doc

    if date := _MOTIF_DATE.search(nom):
        annee, mois = date.groups()
        meta["date_doc"] = f"{annee}-{mois}" if mois else annee

    return meta


class FastEmbedEmbeddings(Embeddings):
    """Embeddings ONNX via `fastembed` : même modèle que la voie PyTorch, sans PyTorch.

    Le modèle est chargé une fois par processus (~10 s) et réutilisé : Streamlit
    recharge le module à chaque interaction, le cache de classe évite de repayer.
    """

    _modeles: ClassVar[dict[str, object]] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _modele(self):
        if self.model_name not in self._modeles:
            from fastembed import TextEmbedding

            self._modeles[self.model_name] = TextEmbedding(self.model_name)
        return self._modeles[self.model_name]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vecteur.tolist() for vecteur in self._modele().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(cfg: Settings = settings) -> Embeddings:
    """Retourne le modèle d'embeddings selon le provider configuré.

    Attention : un index construit avec un provider doit être interrogé avec le
    même — deux implémentations d'un même modèle ne produisent pas des vecteurs
    strictement identiques.
    """
    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.openai_embedding_model)

    if cfg.embedding_provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=cfg.hf_embedding_model)

    return FastEmbedEmbeddings(cfg.hf_embedding_model)


def _load_pdf(path: Path, meta: dict[str, str]) -> list[Document]:
    """Un Document par page, afin de pouvoir citer le numéro de page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # page image-only : rien à indexer
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number, **meta},
            )
        )
    return docs


def _load_text(path: Path, meta: dict[str, str]) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": path.name, **meta})]


def load_documents(source_dir: Path) -> list[Document]:
    """Charge tous les PDF/TXT/MD d'un répertoire, arborescence comprise.

    `metadata["source"]` porte le nom de fichier court : c'est ce qui apparaît
    dans les citations affichées à l'utilisateur. S'y ajoutent les métadonnées
    de provenance déduites du chemin (voir `extraire_metadonnees`).
    """
    docs: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        meta = extraire_metadonnees(path, source_dir)
        docs.extend(_load_pdf(path, meta) if suffix == ".pdf" else _load_text(path, meta))
    return docs


def _splitter(cfg: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_semantique(docs: list[Document], cfg: Settings) -> list[Document]:
    """Découpe sur les sections du contrat, puis à taille fixe à l'intérieur.

    Les `Document` arrivent page par page ; on les regroupe par fichier pour que
    les sections traversant un saut de page restent entières.
    """
    pages_par_source: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        pages_par_source[doc.metadata.get("source", "")].append(doc)

    splitter = _splitter(cfg)
    chunks: list[Document] = []

    for pages in pages_par_source.values():
        pages = sorted(pages, key=lambda d: d.metadata.get("page", 0))
        # Métadonnées communes au fichier : elles valent pour tous ses chunks.
        base = {k: v for k, v in pages[0].metadata.items() if k != "page"}

        for section in decouper_en_sections([p.page_content for p in pages]):
            # Le titre est rappelé en tête de chaque morceau : un passage extrait
            # de son contrat doit rester compréhensible seul. La page citée est
            # celle où le morceau commence réellement, pas celle de la section.
            position = 0
            for morceau in splitter.split_text(section.contenu):
                debut = section.contenu.find(morceau[:60], position)
                if debut >= 0:
                    position = debut
                meta = {
                    **base,
                    "page": section.page_a_l_offset(position),
                    "section": section.titre,
                }
                chunks.append(Document(page_content=f"{section.titre}\n{morceau}", metadata=meta))

    return chunks


def split_documents(docs: list[Document], cfg: Settings = settings) -> list[Document]:
    """Découpe les documents selon la stratégie configurée."""
    if cfg.chunking_strategy == "semantic":
        return _split_semantique(docs, cfg)
    return _splitter(cfg).split_documents(docs)


def get_vectorstore(cfg: Settings = settings) -> Chroma:
    """Ouvre (ou crée) le vector store persistant."""
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=get_embeddings(cfg),
        persist_directory=cfg.persist_directory,
    )


def reset_vectorstore(cfg: Settings = settings) -> None:
    """Vide la collection avant réindexation.

    Sans cela, une seconde ingestion empile les mêmes chunks au lieu de les
    remplacer : l'index gonfle et les mêmes passages ressortent en double.
    """
    store = get_vectorstore(cfg)
    ids = store.get(include=[])["ids"]
    if ids:
        store.delete(ids=ids)


def ingest(source_dir: Path | None = None, cfg: Settings = settings, reset: bool = True) -> int:
    """Pipeline complet d'ingestion. Retourne le nombre de chunks indexés.

    `reset=True` (défaut) reconstruit l'index à partir de zéro, ce qui rend
    l'opération idempotente.
    """
    source_dir = source_dir or cfg.raw_dir
    docs = load_documents(source_dir)
    if not docs:
        return 0
    chunks = split_documents(docs, cfg)
    if reset:
        reset_vectorstore(cfg)
    get_vectorstore(cfg).add_documents(chunks)
    return len(chunks)


def ingest_documents(docs: list[Document], cfg: Settings = settings) -> int:
    """Ingestion de documents déjà chargés (upload Streamlit)."""
    if not docs:
        return 0
    chunks = split_documents(docs, cfg)
    get_vectorstore(cfg).add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"{n} chunks indexés dans {settings.persist_directory}")
