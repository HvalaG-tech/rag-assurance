"""Découpage sémantique des documents contractuels.

Les conditions générales d'assurance françaises ne sont pas structurées en articles
numérotés (relevé sur le corpus : `docs/CORPUS.md`). Elles s'articulent en sections
titrées en majuscules — « LA GARANTIE VOL », « DÉFENSE PÉNALE ET RECOURS » — et en
numérotations hiérarchiques. Ce module découpe d'abord sur ces frontières, afin qu'un
passage envoyé au LLM ne soit jamais amputé du milieu d'une garantie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Un titre de section, tel qu'il apparaît dans les CG : majuscules, éventuellement
# précédé d'une numérotation ("2. VOS GARANTIES", "TITRE 2", "CHAPITRE 1 -").
_MOTIF_MAJUSCULES = re.compile(
    r"^[0-9IVX]{0,4}[\.\-–)\s]*[A-ZÉÈÀÙÇÊÎÔÂÏËÜÖŒ][A-ZÉÈÀÙÇÊÎÔÂÏËÜÖŒ0-9\s'’\-–«»,\.()/&:]{3,79}$"
)

# Structures alternatives, plus rares dans le corpus mais sans ambiguïté.
_MOTIF_ARTICLE = re.compile(r"^\s*article\s+\d+", re.IGNORECASE)
_MOTIF_MARKDOWN = re.compile(r"^#{1,4}\s+\S")
_MOTIF_NUMEROTATION = re.compile(r"^\s*\d{1,2}(\.\d{1,2}){1,3}[\.\s]+\S")

# --- Bruit typographique relevé sur le corpus réel ---
# Pagination : « 10/56 », « - 12 - », « 7 ».
_BRUIT_PAGINATION = re.compile(r"^[\s\-–]*\d{1,4}\s*(/\s*\d{1,4})?[\s\-–]*$")
# Entrée de sommaire : un intitulé suivi du numéro de page — « EXCLUSIONS GÉNÉRALES 26 ».
_BRUIT_SOMMAIRE = re.compile(r"[A-ZÉÈÀÙÇ]\s*[\.\s]{1,}\d{1,3}$")
# Coordonnées postales de l'assureur — « BP 290 », « TSA 46 307 », « CS 70001 ».
_BRUIT_ADRESSE = re.compile(r"^\s*(BP|TSA|CS|CEDEX)\b", re.IGNORECASE)

LONGUEUR_TITRE_MAX = 80
LONGUEUR_TITRE_MIN = 4

# Les conditions générales composent des clauses entières en majuscules (exclusions,
# mentions légales). Leurs lignes ressemblent à des titres ; leur longueur les trahit.
MOTS_TITRE_MAX = 12

# Au-delà de cette proportion de pages, une ligne identique est un en-tête ou un
# pied de page, jamais un titre de section. Le seuil est bas : sur le corpus, les
# en-têtes manquent parfois sur les pages de garde et de sommaire.
SEUIL_ENTETE = 0.3


@dataclass(frozen=True)
class Section:
    """Une section du document, avec la page où elle commence.

    `pages` donne la page de chaque ligne de `contenu` : une section longue
    traverse plusieurs pages, et un passage cité doit pointer sur la bonne.
    """

    titre: str
    contenu: str
    page: int
    pages: tuple[int, ...] = ()

    def page_a_l_offset(self, offset: int) -> int:
        """Page de la ligne qui contient la position `offset` dans `contenu`."""
        if not self.pages:
            return self.page
        ligne = self.contenu.count("\n", 0, max(offset, 0))
        return self.pages[min(ligne, len(self.pages) - 1)]


def est_titre_candidat(ligne: str) -> bool:
    """Vrai si la ligne ressemble à un titre de section.

    Le filtrage du bruit passe avant la reconnaissance : une entrée de sommaire
    ressemble à s'y méprendre au titre qu'elle référence.
    """
    ligne = ligne.strip()

    if not (LONGUEUR_TITRE_MIN <= len(ligne) <= LONGUEUR_TITRE_MAX):
        return False

    if _MOTIF_MARKDOWN.match(ligne):
        return True

    if (
        _BRUIT_PAGINATION.match(ligne)
        or _BRUIT_SOMMAIRE.search(ligne)
        or _BRUIT_ADRESSE.match(ligne)
    ):
        return False

    # Un titre porte des lettres : « 2. 3. 4. » n'en est pas un.
    if sum(c.isalpha() for c in ligne) < 3:
        return False

    if len(ligne.split()) > MOTS_TITRE_MAX:
        return False

    return bool(
        _MOTIF_ARTICLE.match(ligne)
        or _MOTIF_NUMEROTATION.match(ligne)
        or _MOTIF_MAJUSCULES.match(ligne)
    )


def detecter_titres_repetes(pages: list[str], seuil: float = SEUIL_ENTETE) -> set[str]:
    """Lignes candidates présentes sur une forte proportion des pages.

    Ce sont les en-têtes et pieds de page. Sans ce filtre, le document serait
    découpé à chaque page — « ASSURANCE AUTO » figure sur les 38 pages du
    document Allianz.
    """
    if len(pages) < 3:
        return set()

    compte: dict[str, int] = {}
    for page in pages:
        # Une ligne comptée une fois par page, même si elle y figure plusieurs fois.
        for ligne in {ligne.strip() for ligne in page.split("\n") if est_titre_candidat(ligne)}:
            compte[ligne] = compte.get(ligne, 0) + 1

    minimum = max(2, seuil * len(pages))
    entetes = {ligne for ligne, n in compte.items() if n >= minimum}

    # L'extraction PDF fragmente les en-têtes : le document santé d'AXA produit
    # « MA SANTÉ » sur 40 pages et « COMPLÉMENTAIRE SANTÉ MA SANTÉ » sur 9 autres.
    # La variante longue passe sous le seuil alors qu'il s'agit du même en-tête.
    # Seconde passe : toute ligne répétée qui englobe un en-tête déjà identifié.
    variantes = {
        ligne
        for ligne, n in compte.items()
        if n >= 2 and ligne not in entetes and any(e in ligne for e in entetes)
    }

    return entetes | variantes


def decouper_en_sections(pages: list[str]) -> list[Section]:
    """Découpe un document, page par page, sur ses frontières de sections.

    Le texte précédant le premier titre est conservé dans une section « (préambule) » :
    aucune information ne doit disparaître au découpage.
    """
    entetes = detecter_titres_repetes(pages)

    sections: list[Section] = []
    titre_courant = "(préambule)"
    page_courante = 0
    lignes_courantes: list[tuple[str, int]] = []

    def cloturer() -> None:
        if not lignes_courantes:
            return
        contenu = "\n".join(ligne for ligne, _ in lignes_courantes)
        if contenu.strip():
            sections.append(
                Section(
                    titre=titre_courant,
                    contenu=contenu,
                    page=page_courante,
                    pages=tuple(page for _, page in lignes_courantes),
                )
            )

    for numero_page, page in enumerate(pages):
        for ligne in page.split("\n"):
            nette = ligne.strip()
            if not nette or nette in entetes:
                continue
            if est_titre_candidat(nette):
                cloturer()
                titre_courant = nette
                page_courante = numero_page
                lignes_courantes = []
            else:
                lignes_courantes.append((nette, numero_page))

    cloturer()
    return sections
