# Ask your insurance data

**Vos gestionnaires passent vingt minutes à chercher une règle dans trois cents pages de
conditions générales. Ce système répond en dix secondes — avec le passage exact qui fonde
la réponse, et un aveu clair quand le corpus ne la contient pas.**

![Démonstration](docs/demo.gif)

## ▶ [Essayer la démo en ligne](https://rag-assurance.streamlit.app)

*Six questions guidées, sans clé ni inscription. Les questions libres utilisent votre
propre clé API.*

---

## Le problème

Un gestionnaire sinistres, un chargé de conformité ou un conseiller retrouve chaque jour
la même chose : *un* délai, *un* plafond, *une* exclusion — enfouis dans des contrats de
40 à 100 pages, rédigés différemment d'un assureur à l'autre. La recherche plein texte
échoue sur les reformulations ; un assistant IA générique répond avec assurance… et
invente un montant sur trois.

En assurance, une réponse plausible mais fausse coûte plus cher qu'une absence de réponse.

## Ce que ça fait

- **Répond en langage naturel** à partir de conditions générales réelles — quatre contrats
  (auto, habitation, santé, RC vie privée) de quatre assureurs.
- **Cite le passage exact** : assureur, document, page, section, avec la phrase reprise
  surlignée dans l'extrait.
- **S'abstient explicitement** quand le corpus ne permet pas de répondre — 7 fois sur 7 en
  évaluation, sans jamais inventer.
- **Affiche un indice de confiance** fondé sur la qualité de la recherche *et* sur
  l'auto-évaluation du modèle, en retenant le plus prudent des deux.
- **Mesure ce qu'il promet** : 40 questions annotées, six configurations comparées,
  chiffres publiés — y compris ceux qui ne flattent pas.

## Résultats mesurés

Évaluation sur 40 questions écrites à la main contre le corpus (25 factuelles, 8 de
synthèse, 7 sans réponse dans le corpus). Détail et méthode : [docs/EVALUATION.md](docs/EVALUATION.md).

| Configuration de recherche | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| vectoriel seul | 15 % | 27 % | 0,24 |
| BM25 seul | 59 % | 75 % | 0,69 |
| hybride (RRF) | 39 % | 65 % | 0,51 |
| **hybride + reranker** | **62 %** | **82 %** | **0,71** |

| Réponse (Claude Opus 5, hybride + reranker) | |
|---|---:|
| Justesse sur les questions à réponse | **82 %** |
| Abstention correcte sur les questions hors corpus | **7 / 7** |
| Fidélité aux extraits (juge LLM) | **100 %** |
| Coût par question | 0,032 $ |

Les 18 % de questions manquées le sont à la recherche, pas à la rédaction : sur chacune,
le passage attendu n'était pas parmi les cinq transmis, et le système l'a dit plutôt que
de combler.

## Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │                     INGESTION (hors ligne)                    │
  PDF publics ──►│ pypdf ─► métadonnées ─► découpage sémantique ─► embeddings   │──► ChromaDB
  4 contrats     │         (branche,        (sections du          ONNX,         │   761 chunks
                 │          assureur)        contrat)              multilingue   │   9,8 Mo, versionné
                 └──────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────────────────────────────────────────┐
                 │                     REQUÊTE (en ligne)                        │
  question ─────►│ vectoriel ─┐                                                  │
                 │            ├─► fusion RRF ─► reranker* ─► 5 passages ─► LLM ──┼──► réponse sourcée
                 │ BM25 ──────┘                                                  │    ou abstention
                 └──────────────────────────────────────────────────────────────┘    + confiance
                                        * cross-encoder, local uniquement
```

Le fonctionnement complet, module par module, et les raisons de chaque choix :
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Stack

| Composant | Choix | Pourquoi |
|---|---|---|
| LLM | Claude Opus 5 (`langchain-anthropic`), OpenAI commutable | qualité de lecture des clauses ; `effort` comme levier de coût |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` via **fastembed** (ONNX) | multilingue, sans PyTorch : déployable sur l'hébergement gratuit |
| Recherche lexicale | `rank-bm25` | les termes exacts de l'assurance |
| Fusion | Reciprocal Rank Fusion | combine deux classements sans calibrer les scores |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | convertit le rappel en précision (+17 pts recall@5) |
| Vector store | ChromaDB, index versionné | aucun embedding de document au démarrage |
| Interface | Streamlit | conversation, sources surlignées, parcours guidé |
| Qualité | pytest (66 tests), ruff | dont un test qui exécute l'application |

## Installation

```bash
git clone https://github.com/HvalaG-tech/rag-assurance && cd rag-assurance
python -m venv .venv && .venv\Scripts\activate       # Windows ; source .venv/bin/activate ailleurs
pip install -r requirements.txt && copy .env.example .env    # renseigner ANTHROPIC_API_KEY
streamlit run app/main.py
```

L'index est versionné : rien à construire. Pour retélécharger les contrats, relancer une
indexation ou l'évaluation : `python -m scripts.fetch_corpus`, `python -m scripts.build_index`,
`python -m eval.run_eval --compare`.

## Corpus

Quatre conditions générales publiques, téléchargées depuis les sites des assureurs
(provenance et statut de diffusion : [docs/CORPUS.md](docs/CORPUS.md)) :

| Branche | Assureur | Document | Pages |
|---|---|---|---:|
| Auto | Allianz (AllSecur) | Dispositions générales | 38 |
| Habitation | MMA | CG n° 651 o, janv. 2016 | 56 |
| Santé | AXA | CG Complémentaire santé, déc. 2020 | 52 |
| RC vie privée | SMACL | CG modèle 5, févr. 2023 | 16 |

## Limites connues

- **Corpus de démonstration.** Quatre contrats, dont un MMA de 2016 — le site MMA bloque
  le téléchargement automatisé des éditions récentes. Aucune vocation de conseil ni de
  comparaison commerciale.
- **Pas d'OCR.** Un PDF scanné serait ignoré (aucun dans ce corpus).
- **Tableaux aplatis.** L'extraction de texte perd la structure des tableaux de garanties ;
  un montant peut perdre sa colonne.
- **Reranker désactivé en ligne**, pour tenir dans la mémoire de l'hébergement gratuit.
  Les six questions du parcours en profitent (réponses pré-calculées avec) ; les questions
  libres tournent en hybride nu, avec la précision inférieure mesurée ci-dessus.
- **40 questions d'évaluation.** Assez pour comparer des configurations, pas pour affirmer
  une précision au point près ; un écart de 3 points est du bruit.
- **Pas de mémoire de conversation** : chaque question est traitée seule.

## Auteur

**Guillaume Hvala** — Data Scientist indépendant, IA appliquée ([Hvaloa](https://hvaloa.com)).
Sept ans de données, dont quatre sur les sinistres et les leads chez Allianz : les
questions de démonstration sont celles que les équipes métier posent réellement.

Licence MIT.
