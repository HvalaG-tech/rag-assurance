"""Capture les écrans de démonstration du README.

Lance l'application en mode démo, clique sur les questions du parcours et
photographie deux moments : une réponse sourcée, et l'abstention.

Prérequis : l'application doit tourner (`streamlit run app/main.py`), et
`playwright` doit être installé (`pip install playwright && playwright install chromium`).

Usage : python -m scripts.capture_ecrans [url] [dossier_sortie]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL_DEFAUT = "http://localhost:8503"
SORTIE_DEFAUT = Path(__file__).resolve().parent.parent / "docs"

# Le bouton est repéré par un fragment de sa question : les index changent dès
# qu'on ajoute un bouton ailleurs dans la page (barre latérale comprise).
ECRANS = [
    ("déclarer le vol de mon véhicule", "demo-reponse-sourcee.png", True),
    ("capital est versé aux bénéficiaires", "demo-abstention.png", False),
]


def capturer(url: str, sortie: Path) -> None:
    sortie.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        navigateur = p.chromium.launch()
        # Viewport haut : tout tient sans scroll. Streamlit scrolle dans un conteneur
        # interne dont `document.body` a une hauteur nulle, ce qui met en échec
        # `scroll_into_view_if_needed` et les captures d'élément.
        page = navigateur.new_page(
            viewport={"width": 1400, "height": 2000}, device_scale_factor=2
        )

        for fragment, nom, _ in ECRANS:
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(3000)

            # `force` : le bouton est visible et stable, seul le calcul de viewport
            # de Playwright se trompe à cause de ce conteneur.
            page.locator('[data-testid="stButton"] button', has_text=fragment).first.click(
                force=True
            )
            # La réponse est pré-calculée : l'attente couvre le rendu, pas un appel réseau.
            page.wait_for_timeout(5000)

            # L'infobulle du bouton reste affichée tant qu'il a le focus (elle est
            # liée au focus, pas au survol) et masque le titre du parcours.
            page.mouse.move(5, 5)
            page.evaluate("document.activeElement && document.activeElement.blur()")
            page.wait_for_timeout(1500)

            # Rogner sur le contenu réel, pour ne pas photographier du blanc.
            boite = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
            clip = None
            if boite:
                clip = {
                    "x": 0,
                    "y": 0,
                    "width": 1400,
                    "height": min(boite["y"] + boite["height"] + 30, 2000),
                }
            page.screenshot(path=str(sortie / nom), clip=clip)
            print(f"  ecrit {sortie / nom}")

        navigateur.close()


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else URL_DEFAUT
    sortie = Path(sys.argv[2]) if len(sys.argv) > 2 else SORTIE_DEFAUT
    print(f"Capture depuis {url}")
    capturer(url, sortie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
