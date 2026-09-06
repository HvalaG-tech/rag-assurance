# Évaluation — ce qui est mesuré, et ce que ça vaut

> Chiffres du 6 septembre 2026, corpus de 4 contrats (162 pages, 761 chunks), 40 questions.
> Reproduire : `python -m eval.run_eval --compare` (recherche, gratuit) puis
> `python -m eval.run_eval --llm --judge` (génération + juge, ≈ 1,3 $).

## Le jeu de test

40 questions écrites à la main contre le corpus, dans `eval/questions.jsonl` :

| catégorie | nombre | ce qu'elle vérifie |
|---|---:|---|
| factuelle | 25 | un délai, un montant, une condition — dans un seul contrat |
| synthèse | 8 | croiser plusieurs passages, parfois plusieurs contrats |
| **hors corpus** | **7** | **le système dit « information absente » au lieu d'inventer** |

Chaque question à réponse porte une **citation exacte** du contrat (la preuve que la
recherche doit remonter) et les éléments que la réponse doit contenir. Les sept questions
hors corpus sont la partie critique : elles mesurent la promesse commerciale.

Toutes les citations ont été vérifiées dans les documents avant d'être annotées. Les
questions que le système rate sont de vraies questions ratées, pas des annotations
douteuses — chaque preuve existe bien dans l'index (1 à 2 chunks porteurs).

## Étage 1 — la recherche, six configurations

Chaque axe est isolé pour chiffrer sa contribution. `recall@k` : part des preuves
attendues retrouvées dans les k premiers passages (questions à réponse). MRR : 1 / rang
de la première preuve.

| Configuration | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| fixed · vector | 15 % | 24 % | 33 % | 0,24 |
| fixed · hybrid | 36 % | 49 % | 72 % | 0,49 |
| semantic · vector | 15 % | 24 % | 27 % | 0,24 |
| semantic · bm25 | 59 % | 72 % | 75 % | 0,69 |
| semantic · hybrid | 39 % | 52 % | 65 % | 0,51 |
| **semantic · hybrid · rerank** | **62 %** | **72 %** | **82 %** | **0,71** |

Ce que le tableau dit :

- **Le vectoriel seul est faible sur ce corpus** (recall@5 ≈ 30 %). Les questions
  d'assurance portent sur des termes exacts — noms de garantie, délais, montants — que
  l'embedding sémantique ne privilégie pas.
- **BM25 seul est étonnamment fort** (75 %) — pour la même raison. Il reste aveugle aux
  reformulations.
- **L'hybride sans reranker fait moins bien que BM25 seul** en précision (65 % contre
  75 % à k=5). La fusion RRF augmente le rappel (le bon passage est dans les 20 candidats)
  mais un consensus médiocre des deux moteurs l'emporte sur l'excellence d'un seul. C'est
  une propriété mathématique du RRF, vérifiée par un test paramétré — pas un défaut de
  réglage.
- **Le reranker convertit ce rappel en précision** : +17 points de recall@5 et +23 points
  de recall@1 par rapport à l'hybride nu. C'est l'étape qui justifie les deux précédentes.
- **Le découpage sémantique ne change pas le rappel vectoriel** (27 % contre 33 %) ; son
  apport est ailleurs — des passages entiers, un titre de section en tête de chaque chunk,
  une page exacte — ce que mesure l'étage suivant, pas celui-ci.

Les 18 % de questions non retrouvées à k=5 sont concentrées sur les comparaisons à deux
sources (`q026`, `q033` : il faut *les deux* preuves dans les cinq passages) et sur des
formulations éloignées du texte (`q006` renonciation, `q017` délai catastrophe naturelle,
`q024` transport sanitaire à l'étranger).

## Étage 2 — la réponse, configuration de référence

`semantic · hybrid · rerank`, Claude Opus 5, `effort=medium`, 5 passages transmis.

| métrique | valeur | définition |
|---|---:|---|
| justesse | **82 %** | réponses aux 33 questions à réponse contenant les éléments attendus |
| abstention correcte | **7 / 7** | questions hors corpus déclarées « information absente » |
| fausses abstentions | 4 | questions à réponse sur lesquelles le système s'est abstenu |
| fidélité (juge LLM) | **100 %** | aucune affirmation non appuyée par les extraits, sur 40 réponses |
| coût | 0,032 $ / question | ≈ 2 800 tokens en entrée, 700 en sortie |

Lecture :

- **Le système n'a jamais inventé.** Les 7 questions hors corpus ont toutes reçu
  « Information absente du corpus », et le juge n'a relevé aucune affirmation infondée
  sur les 40 réponses.
- **Les 4 fausses abstentions sont toutes des échecs de recherche**, pas de rédaction :
  sur chacune, la preuve n'était pas dans les 5 passages transmis (`recall@5 = 0`). Le
  modèle n'avait pas de quoi répondre et l'a dit — c'est exactement le comportement voulu.
  Améliorer ces cas passe par la recherche (voir « Pistes »), pas par le prompt.
- **Les 6 questions injustes** = ces 4 abstentions + 2 réponses qui ne contiennent pas
  l'élément attendu parce que le passage manquait (`q024`, `q026`).

## Ce que la première exécution a appris

La première passe donnait 79 % de justesse et **6** fausses abstentions, dont deux
(`q001`, `q026`) *avec* la preuve dans les passages transmis. Lecture des réponses : le
modèle lisait « AllSecur » dans l'extrait, « Allianz » dans la question, et concluait à
un document hors sujet — puis donnait la bonne réponse « à titre indicatif » après s'être
abstenu.

AllSecur est la marque directe d'Allianz ; l'écart n'était que de nom. Mais l'en-tête des
extraits ne transmettait que le nom de fichier. Il porte désormais l'assureur, la branche
et la section (`format_context`), et le prompt demande de s'y fier. Résultat : les deux
fausses abstentions disparaissent, la justesse passe à 82 %, la fidélité à 100 %.

Enseignement plus général : **la rigueur exigée du modèle suppose qu'on lui donne de
quoi être rigoureux.** Un système qui s'abstient au moindre doute est bien réglé ; il
faut alors que le contexte lève les doutes légitimes.

## Ce que ces chiffres ne disent pas

- 40 questions suffisent à comparer des configurations, pas à affirmer une précision au
  point près. Un écart de 3 points est du bruit.
- La justesse par règles vérifie la présence d'éléments attendus ; elle ne note pas la
  qualité rédactionnelle ni la complétude d'une synthèse.
- Le juge de fidélité est le même modèle que le générateur (à effort réduit) : il partage
  ses angles morts.
- Le coût est celui de `claude-opus-5` à effort `medium` ; il inclut les tokens de
  raisonnement.

## Pistes

Par ordre de rendement attendu, pour les 18 % de recherche manquée :

1. **Plus de passages pour les questions de synthèse** — `top_k` adaptatif (8 quand la
   question cite plusieurs contrats) ; les deux comparaisons manquées exigeaient deux
   preuves en cinq passages.
2. **Réécriture de requête** — une reformulation par le modèle avant BM25 (« renonciation »
   → « droit de renonciation vente à distance ») pour les questions éloignées du texte.
3. **Reranker en ligne** — via un cross-encoder ONNX (`fastembed` en propose) pour
   retrouver en démo la précision mesurée ici.
