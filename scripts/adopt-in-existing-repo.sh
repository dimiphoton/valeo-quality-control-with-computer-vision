#!/usr/bin/env bash
# Usage : ./scripts/adopt-in-existing-repo.sh /chemin/vers/vieux-repo
#
# À lancer depuis le dossier du template (celui-ci). Ajoute les règles
# Cursor et la structure portfolio à un repo existant, sur une branche
# dédiée — ne touche jamais à main directement, n'écrase jamais un fichier
# de contenu déjà présent (README, pyproject.toml, ROADMAP déjà rempli...).
set -e

TEMPLATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$1"

if [ -z "$TARGET" ]; then
  echo "Usage : ./scripts/adopt-in-existing-repo.sh /chemin/vers/vieux-repo"
  exit 1
fi

if [ ! -d "$TARGET/.git" ]; then
  echo "Erreur : $TARGET n'est pas un repo git (pas de dossier .git)."
  exit 1
fi

cd "$TARGET"

if [ -n "$(git status --porcelain)" ]; then
  echo "Erreur : ce repo a des changements non commités."
  echo "Commit ou stash d'abord, puis relance le script."
  exit 1
fi

BRANCH="feature/mise-en-place-portfolio"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"

# Règles Cursor : toujours mises à jour avec la version du template
mkdir -p .cursor/rules .cursor/hooks
cp "$TEMPLATE_DIR"/.cursor/rules/*.mdc .cursor/rules/
cp "$TEMPLATE_DIR/.cursor/hooks.json" .cursor/
cp "$TEMPLATE_DIR"/.cursor/hooks/* .cursor/hooks/

# Structure administrative : jamais écrasée si déjà présente dans le repo
cp -n "$TEMPLATE_DIR/ROADMAP.md" .
cp -n "$TEMPLATE_DIR/JOURNAL.md" .
cp -n "$TEMPLATE_DIR/CHANGELOG.md" .
mkdir -p brief
cp -n "$TEMPLATE_DIR"/brief/*.md brief/

mkdir -p pictures/readme pictures/experiments pictures/presentations/photos docs
touch pictures/readme/.gitkeep pictures/experiments/.gitkeep pictures/presentations/.gitkeep pictures/presentations/photos/.gitkeep
cp -rn "$TEMPLATE_DIR"/docs/. docs/

mkdir -p .github/workflows
cp -n "$TEMPLATE_DIR/.github/workflows/build-slides.yml" .github/workflows/

mkdir -p .vscode
cp -n "$TEMPLATE_DIR/.vscode/settings.json" .vscode/

git add -A
git commit -m "ajoute le système de règles Cursor et la structure portfolio"

echo ""
echo "Terminé sur la branche $BRANCH dans $TARGET."
echo "Ouvre Cursor sur ce repo et précise que c'est une reprise de projet existant."
echo "Pense à lancer ./scripts/build-slides.sh une fois les présentations rédigées."
