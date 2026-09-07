# Projet 1 — RAG Assurance : « Ask your insurance data »

Système RAG sur des conditions générales d'assurance publiques : réponse sourcée,
abstention explicite, confiance affichée, qualité mesurée. Vitrine du positionnement
« donnée non structurée + traçabilité » de la démarche freelance.

**Lire d'abord** : `docs/ARCHITECTURE.md` (fonctionnement complet, décisions),
`docs/EVALUATION.md` (chiffres et méthode), `docs/CORPUS.md` (provenance des documents).

## État (6 septembre 2026)

Lots 1 à 6 de la feuille de route terminés, sauf trois gestes qui exigent la main de
Guillaume : enregistrer `docs/demo.gif`, déployer sur Streamlit Community Cloud
(`DEMO_MODE=true`, pas de clé côté serveur), pousser le dépôt public `HvalaG-tech/rag-assurance`.

Chiffres de référence : recall@5 82 %, MRR 0,71, justesse 82 %, abstention correcte 7/7,
fidélité 100 %, 0,032 $/question (`eval/results/latest.md`).

## Règles de travail sur ce dépôt

- Python 3.11+, `ruff check` / `ruff format` propres, `pytest` vert avant tout commit.
  Les tests qui exigent l'index ou le réseau s'ignorent d'eux-mêmes.
- TDD : le test d'abord, rouge pour la bonne raison, puis l'implémentation.
- Aucun secret dans le code ni dans l'historique. La clé locale est lue depuis
  `ANTHROPIC_API_KEY` (ou `ANTHROPIC_API_KEY_BACKUP`, repli dans `config.py`).
  La clé d'un visiteur ne passe jamais par `os.environ`.
- Après tout changement de corpus, de prompt ou de recherche, relancer dans l'ordre :
  `python -m scripts.build_index` → `python -m scripts.build_demo_answers` →
  `python -m eval.run_eval --llm --judge`, puis versionner `data/chroma/`,
  `data/demo_answers.json`, `eval/results/latest.md`, et mettre `docs/EVALUATION.md`
  et le README en cohérence. **Publier les chiffres réels, même moyens.**
- Un index construit avec un fournisseur d'embeddings s'interroge avec le même
  (`fastembed` pour l'index versionné).
- **Chroma réécrit ses fichiers à la simple ouverture** : `data/chroma/*.bin` et
  `chroma.sqlite3` apparaissent modifiés après chaque `streamlit run` ou `pytest`,
  sans que le contenu logique change. Vérifier le nombre de chunks (761), puis
  `git checkout -- data/chroma` — ne pas commiter ce bruit. Ne jamais faire de
  `git stash` sur ces fichiers tant qu'un serveur Streamlit tourne : Windows
  verrouille les `.bin` et le stash échoue à mi-chemin.
- Arrêter les serveurs de test avant toute opération git :
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object
  { $_.CommandLine -like '*streamlit run*' } | ForEach-Object { Stop-Process -Id
  $_.ProcessId -Force }` (`pkill -f "streamlit run"` ne les attrape pas, leur ligne
  de commande étant `python.exe -m streamlit`).
- `requirements.txt` = démo (sans PyTorch) ; `requirements-local.txt` = + reranker.
- Documents : uniquement des CG publiquement diffusées, jamais un document interne
  d'un ancien employeur. Provenance à consigner dans `docs/CORPUS.md`.

## Ce que ce projet démontre à un prospect

1. La recherche vectorielle seule ne suffit pas sur des contrats (27 % de rappel) ;
   hybride + reranker atteint 82 %. Chiffré, reproductible.
2. Le système sait dire « je ne sais pas » — 7/7 — et n'invente pas (fidélité 100 %).
3. Chaque réponse est auditable : passage, page, section, score, surlignage.
