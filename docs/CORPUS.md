# Corpus de démonstration — provenance et statut de diffusion

> Constitué le 4 septembre 2026. Quatre documents, quatre branches, quatre assureurs, 162 pages.
> **Tous les documents sont librement téléchargeables depuis des sites publics.**
> Aucun document interne, confidentiel ou issu d'une mission client ne figure dans ce corpus.

## Règle appliquée

Seuls sont retenus des documents contractuels que tout particulier peut télécharger sans
authentification : conditions générales, dispositions générales, notices d'information. Ces
documents sont diffusés publiquement en application de l'obligation précontractuelle
d'information (art. L112-2 du Code des assurances) : leur mise à disposition est la finalité
même de leur publication.

## Le corpus

| Fichier | Branche | Assureur | Type | Édition | Pages | Source |
|---|---|---|---|---|---|---|
| `rc/smacl/cg_rc_vie_privee_2023-02.pdf` | RC vie privée | SMACL | CG modèle 5 | févr. 2023 | 16 | [smacl.fr](https://www.smacl.fr/files/documents/cg-responsabilite-civile-vie-privee-modele5-022023.pdf) |
| `auto/allianz/dg_allsecur_auto.pdf` | Auto | Allianz | Dispositions générales | n.c. | 38 | [espaceclient.allianz.fr](https://espaceclient.allianz.fr/pdf/tarification/allsecur/allsecur_dispo_gen.pdf) |
| `sante/axa/cg_complementaire_sante_2020-12.pdf` | Santé | AXA | CG / notice | déc. 2020 | 52 | [axa.fr](https://www.axa.fr/content/dam/axa-fr-convergence/sante-prev-pj/ipid/CG-Complementaire-sante-masante-2020.pdf) |
| `habitation/mma/cg651o_habitation_2016-01.pdf` | Habitation | MMA | CG n° 651 o | janv. 2016 | 56 | [aic-giovannetti.info](https://aic-giovannetti.info/wp-content/uploads/2016/01/MMA-HABITATION-2016-01-COR651-1.pdf) ⚠️ |

**Total : 162 pages, ~573 000 caractères, ~715 chunks estimés.** Texte nativement extractible sur
l'intégralité du corpus (≈ 3 500 caractères par page) — aucun document scanné, aucun besoin d'OCR.

## Pourquoi 162 pages et pas 150

Le plafond visé était de 150 pages. Couvrir les quatre branches avec des **documents entiers**
en coûte 162 : le plus petit document disponible par branche donne 16 + 38 + 52 + 56.
Descendre sous 150 imposerait soit de sacrifier une branche, soit de tronquer un contrat.

Tronquer a été écarté délibérément : un contrat amputé produit des trous de couverture
invisibles, et le système répondrait « information absente » sur des garanties qui existent en
réalité. Or l'abstention est précisément la promesse que ce projet démontre — elle doit
signaler une vraie absence, jamais un artefact de découpage. Le dépassement de 8 % est le prix
de cette cohérence.

## Réserve sur le document MMA ⚠️

`www.mma.fr` bloque le téléchargement automatisé (HTTP 403, protection anti-bot) : les
conditions générales MMA en vigueur — CG n° 410 q d'avril 2022 — n'ont pas pu être récupérées.
Le document MMA du corpus provient donc d'un **miroir tiers** et est une **édition antérieure**
(janvier 2016).

Il reste des conditions générales authentiques et publiquement diffusées, adaptées à une
démonstration technique de recherche documentaire, mais il ne reflète pas l'offre MMA actuelle.
Le corpus n'a aucune vocation de conseil ni de comparaison commerciale — point rappelé dans la
section « Limites connues » du README.

## Structure documentaire réelle

Relevé sur le corpus, à prendre en compte pour le chunking sémantique :

| Document | `Article N` | Titres en majuscules | Numérotation `N.N` |
|---|---|---|---|
| Allianz auto | 7 | 27 | 37 |
| MMA habitation | 0 | 49 | 2 |
| SMACL RC | 0 | 15 | 2 |
| AXA santé | 0 | 9 | 22 |

**Les conditions générales françaises ne sont pas structurées en articles numérotés.** Elles
s'articulent en sections titrées en majuscules (« LA GARANTIE VOL », « DÉFENSE PÉNALE ET
RECOURS ») et en numérotations hiérarchiques. Le découpage doit donc s'appuyer sur ces deux
motifs, et non sur `Article \d+`.

Attention : les titres en majuscules incluent des en-têtes et pieds de page répétés à chaque
page (« ASSURANCE AUTO », « COMPLÉMENTAIRE SANTÉ MA SANTÉ »). Ils doivent être filtrés, sans
quoi le document serait découpé à chaque page.

## Document synthétique conservé

`cg_multirisque_habitation_demo.md` — document fictif rédigé pour les premiers tests du pipeline.
Conservé comme support de test unitaire déterministe, il ne représente aucun contrat réel.

## Reconstituer le corpus

Les PDF ne sont pas versionnés dans le dépôt (voir `.gitignore`).

```bash
python -m scripts.fetch_corpus          # le noyau, 4 documents, 162 pages
python -m scripts.fetch_corpus --all    # + la réserve, 11 documents, 696 pages (usage local)
```

La réserve rassemble sept documents (AXA *Mon Auto* et *Auto Référence*, AXA *Ma Maison*,
Allianz Habitation, MMA CG 410 o, AXA *Ma Santé*, Macif Vie privée) écartés pour la seule
raison du volume. Leurs URL sont conservées dans `scripts/fetch_corpus.py`.
