"""Retélécharge le corpus de démonstration depuis les sites publics des assureurs.

Les PDF ne sont pas versionnés dans le dépôt. Provenance et statut de diffusion
de chaque document : `docs/CORPUS.md`.

Usage : python -m scripts.fetch_corpus
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

# Le corpus est volontairement plafonné à ~150 pages : au-delà, l'index dépasse le
# budget de l'hébergement gratuit. Le noyau ("core") couvre les quatre branches avec
# quatre assureurs distincts, en 162 pages. La réserve ("extended") rassemble les
# documents écartés pour cette seule raison de volume : ils restent téléchargeables
# avec --all pour une exécution locale sur corpus complet (696 pages).
#
# destination relative à data/raw -> (URL source, sélection)
CORPUS: dict[str, tuple[str, str]] = {
    # --- noyau : 162 pages, 4 branches, 4 assureurs ---
    "rc/smacl/cg_rc_vie_privee_2023-02.pdf": (
        "https://www.smacl.fr/files/documents/cg-responsabilite-civile-vie-privee-modele5-022023.pdf",
        "core",
    ),
    "auto/allianz/dg_allsecur_auto.pdf": (
        "https://espaceclient.allianz.fr/pdf/tarification/allsecur/allsecur_dispo_gen.pdf",
        "core",
    ),
    "sante/axa/cg_complementaire_sante_2020-12.pdf": (
        "https://www.axa.fr/content/dam/axa-fr-convergence/sante-prev-pj/ipid/CG-Complementaire-sante-masante-2020.pdf",
        "core",
    ),
    "habitation/mma/cg651o_habitation_2016-01.pdf": (
        "https://aic-giovannetti.info/wp-content/uploads/2016/01/MMA-HABITATION-2016-01-COR651-1.pdf",
        "core",
    ),
    # --- réserve : écartés pour tenir le plafond de pages ---
    "auto/axa/cg_mon_auto_972115J_2026-04.pdf": (
        "https://media.axa.fr/content/dam/axa-fr/image/particuliers/auto/documents-informations/auto/pdf-mon-auto-cg.pdf",
        "extended",
    ),
    "auto/axa/cg_auto_reference_180209C_2021-11.pdf": (
        "https://media.axa.fr/content/dam/axa-fr/image/particuliers/auto/documents-informations/malus/180209C-1121-version-finale.pdf",
        "extended",
    ),
    "habitation/axa/cg_ma_maison_2025-10.pdf": (
        "https://www.axa.fr/content/dam/axa-fr-convergence/habitation/ipid/Ma_Maison_CG.pdf",
        "extended",
    ),
    "habitation/allianz/dg_allianz_habitation_COM16258.pdf": (
        "https://espaceclient.allianz.fr/pdf/tarification/COM16258.pdf",
        "extended",
    ),
    "habitation/mma/cg410o_habitation_2015-06.pdf": (
        "https://www.resilier.fr/images/conditions-generales-assurance-habitation-mma.pdf",
        "extended",
    ),
    "sante/axa/cg_ma_sante_2026-03.pdf": (
        "https://media.axa.fr/content/dam/axa-fr/image/particuliers/sante/CG-mars-2026.pdf",
        "extended",
    ),
    "rc/macif/cg_multigarantie_vie_privee.pdf": (
        "https://www.macif.fr/files/live/sites/maciffr/files/conditions_generales_habitation/CG_MVPRC.pdf",
        "extended",
    ),
}


def fetch(destination: str, url: str) -> bool:
    """Télécharge un document et vérifie que c'est bien un PDF. Retourne le succès."""
    cible = RAW_DIR / destination
    if cible.exists():
        print(f"  déjà présent  {destination}")
        return True

    cible.parent.mkdir(parents=True, exist_ok=True)
    try:
        reponse = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=60, allow_redirects=True
        )
        reponse.raise_for_status()
    except requests.RequestException as erreur:
        print(f"  ÉCHEC         {destination} — {erreur}")
        return False

    # Une page d'erreur HTML renvoyée en 200 ne doit pas être écrite sur disque.
    if not reponse.content.startswith(b"%PDF"):
        print(f"  ÉCHEC         {destination} — la réponse n'est pas un PDF")
        return False

    cible.write_bytes(reponse.content)
    print(f"  téléchargé    {destination} ({len(reponse.content) // 1024} Ko)")
    return True


def main() -> int:
    # Les consoles Windows sont en cp1252 : les accents casseraient l'affichage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tout = "--all" in sys.argv
    documents = {
        dest: url for dest, (url, selection) in CORPUS.items() if tout or selection == "core"
    }

    libelle = "complet (696 pages)" if tout else "noyau (162 pages)"
    print(f"Corpus de démonstration, {libelle} -> {RAW_DIR}")
    echecs = [dest for dest, url in documents.items() if not fetch(dest, url)]

    print(f"\n{len(documents) - len(echecs)}/{len(documents)} documents disponibles.")
    if not tout:
        print("(--all pour ajouter la réserve, usage local uniquement)")
    if echecs:
        print(
            "Documents manquants (le site source a pu changer d'URL ou bloquer "
            "le téléchargement automatisé) :"
        )
        for dest in echecs:
            print(f"  - {dest} : {CORPUS[dest][0]}")
        print("Les récupérer manuellement depuis un navigateur, ou voir docs/CORPUS.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
