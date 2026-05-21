# Aad2vec

Aad2vec est une application Python/Shiny pour explorer des corpus de discours avec
Word2Vec, des projections ACP, des clusters lexicaux, des cooccurrences et des
mesures de déplacement sémantique entre sous-corpus.

Le projet a été conçu pour travailler sur un corpus de textes électoraux placé dans
`corpus/`, accompagné d'un fichier `metadata.csv` permettant de construire des
partitions comme l'orientation politique, le parti, la date ou le département.

## Fonctionnalités

- segmentation automatique du corpus depuis `corpus/metadata.csv`
- segmentation manuelle depuis l'interface de configuration
- entraînement de modèles Word2Vec par sous-corpus et pour le corpus global
- choix des principales options d'entraînement: dimensions, fenêtre, `min_count`,
  epochs, skip-gram/CBOW, negative sampling, lemmes ou formes graphiques
- visualisation ACP des vocabulaires filtrés par catégorie grammaticale
- clustering hiérarchique des mots projetés
- extraction de voisins sémantiques et de contextes caractéristiques
- projection d'un mot dans plusieurs sous-corpus après alignement Procrustes
- exploration alternative par matrices de cooccurrences
- cache des sorties dérivées dans `outputs/runs/<run>/derived/`

## Structure

```text
.
├── corpus/              # textes source et metadata.csv
├── outputs/             # runs, modèles entraînés et données dérivées
├── src/
│   ├── main.py          # orchestrateur: launcher, entraînement, ouverture UI
│   ├── launcher.py      # choix entre run existant et nouveau run
│   ├── configure_runs.py# configuration des partitions et hyperparamètres
│   ├── app.py           # application Shiny principale
│   ├── ui.py            # interface de visualisation
│   ├── server.py        # logique réactive de l'interface
│   ├── pca_w2v.py       # ACP, clustering, cache des projections
│   ├── semantic_shift.py# alignement et déplacement sémantique
│   ├── cooccurrences.py # projections fondées sur les cooccurrences
│   └── loaders.py       # chargement des runs, modèles et phrases
└── tests/               # tests unitaires
```

## Installation

Le projet a été testé avec Python 3.10.

```bash
python3.10 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Le modèle spaCy français est déclaré dans `requirements.txt`. Si l'installation
depuis l'URL GitHub échoue, il peut être installé séparément:

```bash
python -m spacy download fr_core_news_sm
```

## Lancement

Depuis le dossier `src/`:

```bash
cd src
python main.py
```

L'application ouvre d'abord un écran de choix:

- ouvrir un run déjà présent dans `outputs/runs/`
- créer un nouveau run en configurant la partition du corpus et les paramètres
  d'entraînement

Les nouveaux runs sont enregistrés dans `outputs/runs/` avec leurs modèles,
phrases tokenisées, métadonnées et sorties dérivées.

## Format attendu du corpus

`corpus/` doit contenir des fichiers texte (`.txt`) et peut contenir un
`metadata.csv` séparé par des points-virgules.

La colonne d'identifiant de fichier est détectée automatiquement parmi des noms
comme `filename`, `file`, `fichier`, `nom_fichier`, `document`, `source` ou `id`.
Les autres colonnes peuvent servir à construire des sous-corpus.

Exemple de colonnes:

```text
id;departement-nom;titulaire-soutien;date;orientation
EL009_L_1958_11_007_01_1_PF_01;Ardèche;Union pour la nouvelle République;23/11/1958;UNR
```

## Tests

Les tests utilisent `unittest`, donc aucune dépendance de test additionnelle n'est
nécessaire.

```bash
python -m unittest discover -s tests
```

Depuis l'environnement local du projet:

```bash
./env/bin/python -m unittest discover -s tests
```

## Données générées

`outputs/` et `env/` peuvent devenir volumineux et ne devraient pas être versionnés
dans un dépôt partagé. Les modèles Word2Vec, phrases tokenisées et projections
ACP sont des artefacts reproductibles à partir du corpus et de la configuration
du run.
