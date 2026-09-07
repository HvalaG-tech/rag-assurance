# Comment tout fonctionne

Ce document explique le système de bout en bout : ce que fait chaque module, comment
les données circulent, pourquoi chaque choix a été fait, et comment le faire tourner.
Il s'adresse à quelqu'un qui reprend le projet — ou à un lecteur technique qui veut
vérifier que les promesses du README tiennent.

Les chiffres cités sont ceux de `docs/EVALUATION.md` ; les décisions de corpus sont
détaillées dans `docs/CORPUS.md`.

---

## 1. Le problème, et la forme de la solution

Un gestionnaire sinistres ou un chargé de conformité passe son temps à retrouver *une*
règle — un délai, un plafond, une exclusion — dans des contrats de 40 à 100 pages.
Le système répond à une question en langage naturel en trois temps :

1. **retrouver** les passages du corpus susceptibles de contenir la réponse ;
2. **rédiger** une réponse qui ne s'appuie *que* sur ces passages, en les citant ;
3. **avouer** quand les passages ne permettent pas de répondre, plutôt qu'inventer.

Le troisième temps est le cœur de la proposition de valeur. En assurance, une réponse
plausible mais fausse coûte plus cher qu'une absence de réponse. Tout le système est
conçu pour que l'abstention soit un comportement normal, mesuré, affiché — pas un échec.

```
                 ┌────────────────────────────────────────────────────────────────┐
                 │                        INGESTION (hors ligne)                   │
  PDF publics ──►│ extraction texte ─► métadonnées ─► découpage sémantique ─► ONNX │──► ChromaDB
  (4 contrats)   │  pypdf              (chemin)       (sections du contrat)  embed │   data/chroma/
                 └────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────────────────────────────────────────────┐
                 │                        REQUÊTE (en ligne)                        │
  question ─────►│ vectoriel ─┐                                                    │
                 │            ├─► fusion RRF ─► [reranker] ─► top-k ─► LLM ─► réponse + sources
                 │ BM25 ──────┘                                    │      + abstention
                 │                                                 │      + confiance
                 └─────────────────────────────────────────────────┴──────────────┘
```

---

## 2. Arborescence

```
app/
  config.py        paramètres (Settings), prompts, prix, marqueur d'abstention
  chunking.py      découpage sémantique : détection des sections d'un contrat
  embeddings.py    chargement des PDF, métadonnées, embeddings, ChromaDB, ingestion
  retrieval.py     BM25, fusion RRF, reranking, orchestration de la recherche
  rag_chain.py     appel du LLM, parsing de la réponse, abstention, confiance, coût
  demo.py          parcours guidé, réponses pré-calculées (mode démo)
  main.py          interface Streamlit
eval/
  questions.jsonl  40 questions annotées (vérité terrain)
  metrics.py       recall@k, MRR, justesse, abstention — sans LLM
  judge.py         juge LLM de fidélité
  run_eval.py      harnais : compare les configurations, écrit eval/results/
scripts/
  fetch_corpus.py  retélécharge les PDF depuis les sites publics des assureurs
  build_index.py   construit l'index versionné data/chroma/
  build_demo_answers.py  pré-calcule les 6 réponses du parcours (data/demo_answers.json)
  capture_ecrans.py      photographie la démo pour le README (playwright)
tests/             66 tests, dont un qui exécute l'application Streamlit en headless
docs/              CORPUS.md, EVALUATION.md, ce fichier
data/
  raw/             les PDF (non versionnés) — <branche>/<assureur>/<fichier>.pdf
  chroma/          l'index de démonstration (versionné, ~10 Mo)
  demo_answers.json  réponses du parcours guidé (versionné)
```

Chaque module a une responsabilité et une interface étroites. `chunking.py` ne connaît
ni LangChain ni Chroma : il prend des pages de texte et rend des sections. `retrieval.py`
ne connaît pas le LLM. `rag_chain.py` ne connaît pas Streamlit. On peut tester — et
remplacer — chaque étage seul.

---

## 3. Ingestion : des PDF à l'index

### 3.1 Chargement et métadonnées (`embeddings.py`)

`load_documents(data/raw)` parcourt l'arborescence et produit **un `Document` par page**
(`pypdf`), avec pour métadonnées :

| clé | origine | exemple |
|---|---|---|
| `source` | nom du fichier | `dg_allsecur_auto.pdf` |
| `page` | index de page (0-indexé, affiché +1) | `29` |
| `branche` | 1er dossier | `auto` |
| `assureur` | 2e dossier | `allianz` |
| `type_doc` | préfixe du fichier (`cg`, `dg`, `ipid`, `notice`, `faq`) | `DG` |
| `date_doc` | `AAAA-MM` dans le nom | `2023-02` |

Tout élément absent vaut `"inconnu"` : un fichier hors convention s'indexe quand même.
Ces métadonnées servent au **filtrage** (ne chercher que dans la branche santé), à la
**citation** (assureur, page, section) et à l'**en-tête des extraits** envoyés au modèle.

> Point d'attention rencontré : plusieurs CG d'assureurs sont chiffrées en AES
> (protection en écriture). `pypdf` a alors besoin de `cryptography`, d'où sa présence
> dans `requirements.txt`.

### 3.2 Découpage sémantique (`chunking.py`)

Un découpage à taille fixe coupe au milieu d'une garantie ; le passage envoyé au modèle
perd alors son sens. Le découpage sémantique cherche d'abord les **frontières de sections**
du contrat, puis ne re-découpe à taille fixe qu'*à l'intérieur* d'une section trop longue.

Ce que sont réellement ces frontières a été relevé sur le corpus (`docs/CORPUS.md`) :
**les conditions générales françaises ne sont pas structurées en « Article N »**. Elles
s'articulent en titres en majuscules (« LA GARANTIE VOL ») et en numérotations `2.10.3`.
`est_titre_candidat()` reconnaît ces motifs après avoir écarté le bruit typographique :

| bruit | exemple | règle |
|---|---|---|
| pagination | `10/56` | ligne numérique |
| entrée de sommaire | `EXCLUSIONS GÉNÉRALES 26` | titre suivi d'un numéro de page |
| adresse | `TSA 46 307` | préfixes postaux |
| clause en capitales | `SA CHARGE DANS LA MESURE OÙ…` | plus de 12 mots |

`detecter_titres_repetes()` élimine ensuite les **en-têtes et pieds de page** : toute ligne
candidate présente sur plus de 30 % des pages, plus ses variantes fragmentées par
l'extraction PDF (« MA SANTÉ » sur 40 pages et « COMPLÉMENTAIRE SANTÉ MA SANTÉ » sur 9
désignent le même bandeau). Sans ce filtre, chaque page ouvrirait une section.

`decouper_en_sections()` produit des `Section(titre, contenu, page, pages)` — `pages`
donnant la page **de chaque ligne**, pour qu'un chunk tiré de la 3e page d'une section
longue cite la 3e page, pas la 1re.

Chaque chunk final porte sa `section` en métadonnée **et** son titre en première ligne :
un passage extrait de son contrat reste compréhensible seul, par le modèle comme par
le lecteur.

`Settings.chunking_strategy` vaut `semantic` (défaut) ou `fixed` — ce dernier n'existe
que pour chiffrer l'apport du premier en évaluation.

### 3.3 Embeddings (`embeddings.py`)

Modèle : `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, multilingue, bon en
français). Trois fournisseurs derrière `get_embeddings()` :

| `EMBEDDING_PROVIDER` | implémentation | quand |
|---|---|---|
| `fastembed` (défaut) | ONNX Runtime, ~220 Mo, sans PyTorch | partout — c'est ce qui rend la démo déployable sur l'hébergement gratuit |
| `hf` | sentence-transformers (PyTorch, ~1 Go) | poste de développement, si besoin |
| `openai` | API | jamais utilisé ici : les documents ne sortent pas |

Règle absolue : **un index construit avec un fournisseur s'interroge avec le même**. Deux
implémentations d'un même modèle ne produisent pas des vecteurs strictement identiques.
L'index versionné a été construit avec `fastembed`.

### 3.4 Vector store et ingestion

ChromaDB, persistant dans `data/chroma/`. `ingest()` est **idempotente** : elle vide la
collection avant de réindexer (`reset_vectorstore`). Sans cela, deux ingestions
successives empilaient les mêmes chunks et les passages ressortaient en double — piège
rencontré et corrigé au Lot 1.

Résultat sur le corpus : **761 chunks, 210 sections, index de 9,8 Mo**. L'index est
versionné (décision 3 de la feuille de route) : la démo en ligne ne calcule aucun
embedding de document au démarrage — seule la *question* est vectorisée, en 60 ms.

---

## 4. Recherche : retrouver le bon passage (`retrieval.py`)

### 4.1 Pourquoi deux moteurs

La recherche vectorielle rapproche des *sens* ; elle échoue sur les **termes exacts** qui
font le quotidien de l'assurance — un nom de garantie, un montant, une référence. Sur le
corpus, interrogée sur « bris de vitres », elle ne ramène pas la section « LA GARANTIE BRIS
DE VITRES » dans ses 20 premiers résultats. BM25, lexical, la place première.

Inversement BM25 ignore les reformulations (« délai pour prévenir l'assureur » ≠ « délai
de déclaration »). D'où la combinaison.

### 4.2 Fusion RRF

Les deux moteurs rendent chacun `fetch_k = 20` candidats. Leurs scores ne sont pas
comparables (cosinus contre BM25) ; la **Reciprocal Rank Fusion** ne regarde que les rangs :

```
score(passage) = Σ sur les classements où il figure de  1 / (k + rang)      k = 60
```

Un passage vu par les deux moteurs cumule deux termes et passe devant un passage vu par
un seul. C'est la propriété recherchée : le **rappel** monte — sur les 5 questions témoins
du Lot 2, l'hybride ne manque jamais le bon passage là où chaque moteur seul le rate.

Limite mesurée, et documentée par un test : à somme de rangs égale, un passage 1er-et-3e
bat toujours un passage 2e-et-2e, quel que soit `k` (`1/(k+r)` est convexe). La fusion
gagne du rappel, pas de la précision : le bon passage est là, mais souvent 4e ou 5e.

### 4.3 Reranking

C'est le rôle du **cross-encoder** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
multilingue) : il lit chaque paire (question, passage) et note si le passage *répond*.
Appliqué aux 20 candidats fusionnés, il remonte le bon passage en tête — sur les
questions témoins : 4→1, 5→2, 18→1.

Il coûte ~1 s par question et tire PyTorch : **désactivé sur la démo en ligne**
(`USE_RERANKER=false`), activé en local, en évaluation et pour pré-calculer les réponses
du parcours de démonstration — les visiteurs profitent de sa précision sans en payer la RAM.

### 4.4 Ce que rend `rechercher()`

Les `top_k` (5) meilleurs passages, chacun enrichi de :

- `score` — RRF, ou logit du cross-encoder si le reranker a tourné (`score_rrf` conservé) ;
- `rangs` — le rang attribué par chaque moteur (`{"vector": 3, "bm25": 1}`).

L'interface s'en sert pour afficher la pertinence en barre et expliquer *pourquoi* un
passage est là. Le filtrage de métadonnées (`filtre={"branche": "sante"}`) s'applique
aux deux moteurs.

L'index BM25 n'est pas persisté : il se reconstruit depuis Chroma en quelques millisecondes
(761 chunks) et reste mémoïsé. Chroma est la seule source de vérité ; aucun second
index ne peut se désynchroniser du premier.

---

## 5. Réponse : rédiger, citer, s'abstenir (`rag_chain.py`)

### 5.1 Le contexte envoyé au modèle

`format_context()` assemble les passages avec un en-tête complet :

```
[Extrait 1 — Source: dg_allsecur_auto.pdf, p.30 — assureur : ALLIANZ · branche : auto · section : DISPOSITIONS EN CAS DE SINISTRE]
DISPOSITIONS EN CAS DE SINISTRE
Vol et tentative de vol, vandalisme 2 jours ouvrés - …
```

L'assureur et la branche dans l'en-tête ne sont pas décoratifs. Sans eux, le modèle
lisait « AllSecur » dans l'extrait, « Allianz » dans la question, et s'abstenait à tort
(AllSecur est la marque directe d'Allianz). C'est un cas d'école : la rigueur demandée au
modèle exige qu'on lui donne de quoi être rigoureux.

### 5.2 Le prompt (`config.py`)

Le prompt système impose : ne répondre qu'à partir des extraits ; se fier à l'en-tête
pour identifier le contrat ; citer `[Source: fichier, p.N]` dans le corps ; reprendre le
vocabulaire contractuel ; et deux conventions **détectables par une règle** :

- quand le corpus ne permet pas de répondre, commencer par la phrase exacte
  `INFORMATION ABSENTE DU CORPUS.` puis dire ce qui manque ;
- terminer par une ligne `CONFIANCE: élevée | moyenne | faible`.

Ces conventions évitent un second appel LLM pour détecter l'abstention ou estimer la
confiance : un `startswith` et une expression régulière suffisent. `est_une_abstention()`
tolère la mise en forme Markdown (le modèle met volontiers le marqueur en gras).

### 5.3 Le modèle

Claude Opus 5 (`claude-opus-5`) via `langchain-anthropic`, avec `output_config.effort`
(`medium` par défaut) comme levier coût/qualité. Les modèles Claude 5 renvoient une
**liste de blocs** (raisonnement + texte) : `texte_de_la_reponse()` n'en garde que le
texte. Le fournisseur OpenAI reste sélectionnable (`LLM_PROVIDER=openai`).

La clé API d'un visiteur (mode démo) est passée **explicitement** au client, jamais
écrite dans `os.environ` : sur un serveur Streamlit partagé, une variable d'environnement
fuirait vers toutes les sessions.

### 5.4 Confiance

Deux signaux, combinés par le **plus prudent** des deux :

| signal | calcul |
|---|---|
| recherche | score du 1er passage — RRF normalisé par son plafond `2/(k+1)`, ou sigmoïde du logit du reranker ; ≥ 0,6 élevée, ≥ 0,3 moyenne, sinon faible |
| modèle | sa propre ligne `CONFIANCE:` |

« Élevée » exige que ni la recherche ni le modèle ne doutent. Un seul « faible » suffit
à afficher « faible ». En assurance, mieux vaut sous-promettre.

### 5.5 Ce que rend `answer()`

Un `RagAnswer` : texte, sources (avec scores et rangs), `abstention`, `confiance`
globale et ses deux composantes, `usage` (tokens) et `cout_estime` (grille de prix dans
`PRIX_PAR_MILLION`). Tout ce qu'il faut pour auditer une réponse est dans l'objet.

---

## 6. Évaluation : mesurer plutôt qu'affirmer (`eval/`)

### 6.1 Le jeu de test

`eval/questions.jsonl` : **40 questions écrites à la main** contre le corpus — 25
factuelles, 8 de synthèse (plusieurs documents), **7 sans réponse dans le corpus**. Ces
sept-là vérifient la promesse centrale : le système doit s'abstenir, pas inventer.

Chaque question à réponse porte une **citation exacte** du contrat (`expected_evidence`)
et les éléments que la réponse doit contenir (`expected_answer_contains`, alternatives
séparées par `|`). La vérité terrain de la recherche est cette citation : un passage est
correct s'il vient du bon fichier *et* la contient — plus robuste qu'un numéro de page ou
un titre de section, sensibles aux aléas du découpage. La comparaison normalise accents,
apostrophes typographiques et espaces (`normaliser()`), car le texte extrait des PDF les
mélange.

### 6.2 Les métriques (`metrics.py`, sans LLM)

| étage | métrique | ce qu'elle mesure |
|---|---|---|
| recherche | recall@1 / @3 / @5 | part des preuves attendues dans les k premiers passages |
| recherche | MRR | 1 / rang de la première preuve trouvée |
| réponse | justesse | la réponse contient les éléments attendus (questions à réponse) |
| réponse | abstention correcte | le système a déclaré l'information absente (questions hors corpus) |
| réponse | fausses abstentions | il s'est abstenu sur une question qui avait une réponse |
| coût | tokens, $ par question | depuis `usage` et la grille de prix |

### 6.3 Le juge (`judge.py`)

La justesse par règles ne voit pas une affirmation juste mais **inventée**. Un juge LLM
(même modèle, `effort=low`) relit chaque réponse face aux extraits et tranche
`FIDELE` / `NON_FIDELE`. Activé par `--judge`, un appel par question.

### 6.4 Le harnais (`run_eval.py`)

```bash
python -m eval.run_eval --compare          # recherche seule, 6 configurations, gratuit
python -m eval.run_eval --llm --judge      # + génération et juge, configuration de référence
python -m eval.run_eval --llm --limit 5    # essai à blanc
```

Les six configurations isolent chaque axe : `fixed`/`semantic` × `vector`/`bm25`/`hybrid`,
plus le reranker. La comparaison des découpages exige un index par stratégie : ils sont
construits dans `data/chroma_eval/` (non versionné), séparés de l'index de démo.

La génération ne tourne que sur la configuration de référence pour contenir le coût
(~1,3 $ pour 40 questions avec juge). Sorties : `eval/results/<date>_<tag>.md` (lisible),
`.json` (détail par question, non versionné) et `latest.md`, que l'interface affiche dans
« Comment ce système est évalué ».

Les chiffres obtenus, et ce qu'ils disent, sont dans `docs/EVALUATION.md`.

---

## 7. Interface (`main.py`, `demo.py`)

Une conversation (`st.chat_input` / `st.chat_message`, historique en session). Chaque
question est indépendante pour la recherche ; seul l'affichage est conversationnel.

**Ce que l'écran montre pour chaque réponse** :

- le texte, avec ses citations `[Source: …]` ;
- un badge de confiance (élevée / moyenne / faible) et ses deux composantes ;
- les passages sources : assureur, branche, fichier, page, section, **barre de
  pertinence**, rang par moteur, et **surlignage** des phrases que la réponse reprend
  (une phrase est surlignée quand ≥ 60 % de ses mots significatifs figurent dans la réponse) ;
- en cas d'abstention, un encart distinct — « Information absente du corpus » — avec les
  passages tout de même consultés. L'abstention est une démonstration, pas un échec.

**Le parcours guidé** : six boutons, à cliquer dans l'ordre, qui racontent une histoire —
factuelle simple → terme exact (l'apport de BM25) → synthèse multi-documents → hors
corpus (l'abstention) → comparaison de contrats → piège sur une exclusion.

**Le mode démo** (`DEMO_MODE=true`) : ces six questions sont servies depuis
`data/demo_answers.json`, pré-calculées par `scripts/build_demo_answers.py` avec le
reranker activé — coût nul, latence nulle, aucune clé. Toute question libre exige la clé
du visiteur (champ masqué, jamais journalisée), avec un plafond par session
(`MAX_QUESTIONS_PAR_SESSION`). Une démo publique branchée sur une clé personnelle est
une facture ouverte ; ce montage est le seul où un prospect voit le système fonctionner
sans rien faire, sans risque pour l'auteur.

Le panneau latéral, replié par défaut, expose le mode de recherche, `top_k`, le reranker,
la clé, et le tableau d'évaluation.

**Quand ça se passe mal.** L'interface ne doit jamais montrer de trace Python à un
visiteur. Deux garde-fous :

- *Avant l'appel* — sans clé du visiteur, la question libre est refusée en mode démo (elle
  ne doit jamais être facturée au propriétaire) et lorsqu'aucune clé n'existe nulle part.
  Le mode démo se déduit d'ailleurs de l'absence de clé serveur, si bien qu'un réglage
  `DEMO_MODE` oublié au déploiement ne peut pas casser la page.
- *Après l'appel* — `message_erreur_api()` traduit l'échec en une phrase actionnable : clé
  refusée, crédit épuisé, droits insuffisants, trop de requêtes, réseau, service surchargé.
  La classification s'appuie sur le texte de l'erreur plutôt que sur les classes du SDK,
  que LangChain enveloppe. Le message brut n'est jamais réaffiché : il cite parfois un
  fragment de la clé (`masquer_les_cles()` sert partout où un texte brut pourrait être
  journalisé). Un échec ne décompte pas de question au visiteur.

`tests/test_app.py` exécute réellement le script (`streamlit.testing.v1.AppTest`) et
vérifie qu'il démarre sans exception, sans appel LLM. Il couvre aussi le parcours d'un
visiteur sans clé, et celui d'une clé refusée.

---

## 8. Faire tourner

```bash
python -m venv .venv && .venv\Scripts\activate         # Windows
pip install -r requirements.txt                          # démo, sans PyTorch
pip install -r requirements-local.txt                    # + reranker (développement)
copy .env.example .env                                   # ANTHROPIC_API_KEY

python -m scripts.fetch_corpus        # les 4 PDF depuis les sites des assureurs
python -m scripts.build_index         # -> data/chroma (déjà versionné : facultatif)
streamlit run app/main.py
```

En ligne de commande : `python -m app.rag_chain "Quel est le délai de prescription ?"`.

Tests et qualité : `pytest` (66 tests, ~30 s ; ceux qui exigent l'index ou le réseau
s'ignorent d'eux-mêmes) et `ruff check` / `ruff format` (configuration dans `pyproject.toml`).

### Déployer la démo (Streamlit Community Cloud)

1. Dépôt GitHub public, `requirements.txt` à la racine (sans PyTorch : `fastembed` suffit).
2. Fichier principal : `app/main.py`. Python 3.11 ou 3.12 dans les paramètres avancés.
3. Secrets côté plateforme : `DEMO_MODE = "true"` — **pas de clé API** : le parcours
   guidé est pré-calculé, les questions libres utilisent la clé du visiteur.
4. Démarrage à froid : ouverture de l'index (~2 s) et chargement du modèle
   d'embeddings ONNX (~10 s), une fois par processus.

Après tout changement de corpus, de prompt ou de recherche : relancer
`build_index`, `build_demo_answers` et `run_eval --llm --judge`, puis versionner
`data/chroma/`, `data/demo_answers.json` et `eval/results/latest.md`.

---

## 9. Les décisions, et pourquoi

| décision | alternative écartée | raison |
|---|---|---|
| Corpus de 4 contrats, 162 pages, 4 assureurs | 11 documents, 696 pages | plafond de l'hébergement gratuit ; couvrir 4 branches avec des documents *entiers* coûte au minimum 162 pages ; tronquer un contrat créerait de fausses absences |
| Découpage sur titres en majuscules | `Article \d+` | relevé sur le corpus : 7 occurrences d'« Article N » sur 4 documents, 0 sur 3 d'entre eux |
| BM25 reconstruit depuis Chroma | index BM25 persisté | source unique de vérité, pas de fichier binaire versionné, quelques ms |
| RRF puis reranker | pondérer les scores | les scores ne sont pas comparables ; le RRF donne le rappel, le reranker la précision |
| `fastembed` par défaut | sentence-transformers | même modèle, sans PyTorch : 220 Mo au lieu d'1 Go, déployable |
| Reranker désactivé en ligne, actif pour le pré-calcul | reranker partout | RAM de l'hébergement gratuit ; les visiteurs profitent quand même de sa précision sur le parcours |
| Abstention par marqueur textuel | second appel « classifieur » | détectable par règle, coût nul, mesurable en évaluation |
| Métadonnées dans l'en-tête des extraits | fichier seul | sans elles, le modèle s'abstenait à tort sur « Allianz » vs « AllSecur » |
| Clé du visiteur passée au client, jamais dans `os.environ` | `os.environ` | un serveur Streamlit partage le processus entre sessions |
| Preuve = citation exacte du contrat | page ou section attendue | robuste aux aléas du découpage ; annotation vérifiable |

---

## 10. Limites connues

- **Corpus de démonstration** : 4 contrats, dont un MMA de 2016 (le site MMA bloque le
  téléchargement automatisé des éditions récentes). Aucune vocation de conseil.
- **Pas d'OCR** : un PDF scanné serait ignoré (aucun dans ce corpus).
- **Tableaux** : l'extraction `pypdf` aplatit les tableaux de garanties ; un montant lu
  dans un tableau peut perdre sa colonne.
- **Reranker absent en ligne** pour raison de mémoire : les questions libres de la démo
  tournent en hybride sans reranking, donc avec la précision inférieure mesurée.
- **Évaluation sur 40 questions** : assez pour comparer des configurations, pas pour
  affirmer une précision au point près.
- **Recherche sans mémoire de conversation** : chaque question est traitée seule.
