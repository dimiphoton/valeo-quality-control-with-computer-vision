# Identité du projet (métier · domaine · stack)

Ce fichier a deux rôles :

1. **Les choix du portfolio** — pourquoi on classe ainsi, et comment
   choisir (ne pas réécrire à chaque repo).
2. **Ce projet** — les trois valeurs à remplir au kickoff. C'est la
   source. README, covers Marp, description GitHub et topics en découlent.
   On ne les invente pas à un autre endroit.

---

## Ce projet

À remplir dès le cadrage, **avant** la roadmap. Une valeur principale
par champ. L'agent propose, l'utilisateur valide.

| Champ | Valeur (FR) | Valeur (EN, pour le README) |
|---|---|---|
| **Métier** | Machine learning | Machine learning |
| **Domaine** | Industrie / contrôle qualité | Industry / quality control |
| **Stack courte** | Python / PyTorch / ONNX / AWS | Python / PyTorch / ONNX / AWS |
| **Ligne identité** | `Machine learning · Industrie / contrôle qualité · Python / PyTorch / ONNX / AWS` | `Machine learning · Industry / quality control · Python / PyTorch / ONNX / AWS` |
| **Topics GitHub** | `machine-learning`, `computer-vision`, `python`, `pytorch`, `quality-control` | |
| **About GitHub** | `Machine learning · Industry · Python / PyTorch / ONNX / AWS` | |

Période / maille des données (pour la cover, en plus de la ligne) :

- Valeo · Challenge Data ENS #157 · 8 278 images train

Une fois validé : recopier dans le tableau du `README.md`, sur la cover
des 4 présentations, puis `gh repo edit` (description + topics) si le
remote existe.

---

## Pourquoi ces trois champs

Un recruteur ouvre le repo 20 secondes. Il doit voir **quel métier ça
illustre**, **dans quel secteur**, **avec quelle stack** — sans lire
l'objectif.

Avant ce bandeau :

- la **stack** était visible (badges), souvent trop longue ;
- le **métier** était noyé dans « Data specialty » (on mélangeait BI et
  SQL, ML et pandas) ;
- le **domaine** (énergie, agri, bâtiment) n'existait que dans le titre.

« Niveau » et « statut » restent utiles plus bas dans le README. Ils ne
remplacent pas le bandeau.

Ne pas confondre :

- **Métier** = le rôle qu'on montre (BI, data engineering, ML, géospatial).
- **Domaine** = le secteur métier (énergie, agriculture, bâtiment…).
- Dans les anciennes règles, « domaine » voulait dire ML vs BI : c'est
  désormais le **métier**.

---

## Listes fermées

Une valeur **principale**. Un second métier se note en une phrase dans
`brief/objectif.md`, pas dans le bandeau.

### Métier

| FR | EN (README) | Topic GitHub |
|---|---|---|
| BI / analyse | BI | `bi` |
| Data engineering | Data engineering | `data-engineering` |
| Machine learning | Machine learning | `machine-learning` |
| Géospatial | Geospatial | `geospatial` |

Comment choisir : *quel poste un recruteur doit cocher en voyant ce
repo ?* Un dashboard Streamlit sur de l'open data wallon → **BI**. Un
entrepôt DuckDB, grains natifs, vues SQL → **Data engineering**. Un
modèle avec baseline et validation → **Machine learning**. Un CRS, une
carte, du géocode → **Géospatial**.

### Domaine

| FR | EN (README) | Topic GitHub |
|---|---|---|
| Énergie | Energy | `energy` |
| Agriculture / climat | Agriculture / climate | `agriculture` ou `climate` |
| Bâtiment | Buildings | `buildings` |
| Mobilité | Mobility | `mobility` |
| Stats publiques | Public statistics | `open-data` |
| Autre | *(écrire le secteur en clair)* | 1 tag court |

Le territoire (Wallonie, Belgique) n'est **pas** le domaine. On le met
dans la cover (`Wallonie · 2000–2024`) et en topic `wallonia` si utile.

### Stack

- **4 ou 5 outils max** dans le bandeau et les badges README. Le
  `pyproject.toml` peut en lister plus.
- Python presque toujours en premier, puis ce qui *distingue* le projet
  (DuckDB, Streamlit, scikit-learn, GeoPandas…).
- Pas de mur `for-the-badge` : un recruteur ne scanne pas 9 logos.

Topics stack typiques : `python`, `duckdb`, `sql`, `streamlit`,
`scikit-learn`, `geopandas`.

**Maximum 5 topics GitHub au total** : 1 métier + 1 domaine + 2 stack +
`wallonia` (ou l'inverse). Pas la peine de tout tagger.

---

## Où ça se propage (automatique)

L'agent ne demande pas « tu veux un bandeau ? ». Dès que `brief/identite.md`
est validé, il met à jour, dans le même geste :

1. **README** — lignes Role / Domain / Stack du tableau sous le titre
   (README en anglais).
2. **Covers Marp** — une ligne de texte sous la question,
   `Métier · Domaine · Stack`, **sans** mur de badges. Même ligne sur
   les 4 decks (c'est de la signalétique, pas le récit). RH : on peut
   omettre la stack si ça surcharge ; métier + domaine suffisent.
3. **GitHub** — si le remote existe :
   `gh repo edit -d "Role · Domain · Stack"` et
   `gh repo edit --add-topic …` pour chaque topic validé.

On ne maintient pas un quatrième fichier d'identité. Celui-ci est la
source ; le README est la vitrine.

---

## Exemples (repos déjà là, pour caler le geste)

- PEB : `BI · Bâtiment · Python / DuckDB`
- Agri-climat : `Data engineering · Agriculture / climat · Python / DuckDB / Streamlit`
- Électricité renouvelable : `BI · Énergie · Python / DuckDB / Streamlit`
