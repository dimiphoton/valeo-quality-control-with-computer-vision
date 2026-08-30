#!/usr/bin/env bash
# Prévisualisation locale OPTIONNELLE des présentations.
#
# Ce n'est plus une étape obligatoire : GitHub Actions
# (.github/workflows/build-slides.yml) régénère automatiquement
# docs/slides/ à chaque push sur main. Lance ce script uniquement si tu
# veux voir le rendu avant de pousser.
#
# Nécessite Node.js (npx est utilisé, aucune installation permanente
# requise).
set -e

npx --yes @marp-team/marp-cli -I docs/presentations/ -o docs/slides/ --allow-local-files --theme-set docs/presentations/portfolio.css

echo ""
echo "Aperçu généré dans docs/slides/ (sera régénéré par CI au push sur main) :"
ls docs/slides/*.html 2>/dev/null || echo "(aucun fichier .html trouvé — vérifie les erreurs ci-dessus)"
