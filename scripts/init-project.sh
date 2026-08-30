#!/usr/bin/env bash
# Usage : ./scripts/init-project.sh nom-du-projet
# À lancer une seule fois, juste après avoir créé un nouveau repo depuis
# le template (bouton "Use this template" sur GitHub), avant le premier
# commit réel.
set -e

if [ -z "$1" ]; then
  echo "Usage : ./scripts/init-project.sh nom-du-projet"
  echo "Exemple : ./scripts/init-project.sh prediction-prix-immo"
  exit 1
fi

SLUG="$1"                                    # ex : prediction-prix-immo
PACKAGE=$(echo "$SLUG" | tr '-' '_')          # ex : prediction_prix_immo
TITLE=$(echo "$SLUG" | tr '-' ' ')            # ex : prediction prix immo

if [ ! -d "src/mon_projet" ]; then
  echo "src/mon_projet introuvable — le script a-t-il déjà été lancé ?"
  exit 1
fi

mv "src/mon_projet" "src/$PACKAGE"
perl -pi -e "s/mon_projet/$PACKAGE/g" pyproject.toml
perl -pi -e "s/mon-projet/$SLUG/g" pyproject.toml
perl -pi -e "s/# Nom du projet/# $TITLE/" README.md

echo "Fait : src/$PACKAGE, pyproject.toml et README.md mis à jour."
echo "Ensuite, dans Cursor : remplir brief/identite.md (métier, domaine, stack),"
echo "puis propager README + covers + topics GitHub."
echo "Tu peux maintenant supprimer scripts/init-project.sh si tu veux."
