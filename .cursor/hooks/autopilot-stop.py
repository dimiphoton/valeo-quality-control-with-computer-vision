"""Hook stop : relance l'agent tant qu'il reste une case ouverte dans ROADMAP.md.

Ne boucle pas pendant le cadrage (étapes « à définir », identité non validée),
ni si l'utilisateur a posé .cursor/autopilot.off ou .cursor/autopilot.pause.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
ROADMAP = ROOT / "ROADMAP.md"
IDENTITE = ROOT / "brief" / "identite.md"
OFF = ROOT / ".cursor" / "autopilot.off"
PAUSE = ROOT / ".cursor" / "autopilot.pause"

UNCHECKED = re.compile(r"^\s*-\s*\[\s*\]\s+(.*)$")
PLACEHOLDER = re.compile(r"à définir", re.IGNORECASE)


def _load_stdin() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _open_items() -> list[str]:
    if not ROADMAP.is_file():
        return []
    items: list[str] = []
    for line in ROADMAP.read_text(encoding="utf-8", errors="replace").splitlines():
        match = UNCHECKED.match(line)
        if match:
            items.append(match.group(1).strip())
    return items


def _identite_pas_validee() -> bool:
    if not IDENTITE.is_file():
        return True
    text = IDENTITE.read_text(encoding="utf-8", errors="replace")
    return "_à valider_" in text or "à valider" in text.split("## Ce projet", 1)[-1][:800]


def _should_continue(payload: dict) -> bool:
    if payload.get("status") != "completed":
        return False
    if OFF.is_file() or PAUSE.is_file():
        return False
    items = _open_items()
    if not items:
        return False
    if all(PLACEHOLDER.search(item) for item in items):
        return False
    if _identite_pas_validee():
        return False
    return True


FOLLOWUP = (
    "Autopilot : enchaîne. Lis ROADMAP.md, brief/identite.md et JOURNAL.md. "
    "Fais uniquement la première case [ ] encore ouverte — pas les suivantes. "
    "Branche feature/<nom-court>, code, tests, mets à jour les présentations "
    "si l'étape est significative, commit sur la branche, merge dans main "
    "selon la règle 03 (sans .cursor sur main), coche la case, journal. "
    "Si tu es bloqué (outil à installer, décision métier, données manquantes, "
    "tests que tu ne sais pas corriger), écris la raison dans "
    ".cursor/autopilot.pause et arrête — ne continue pas. "
    "Ne pousse pas vers origin. Ne redemande pas confirmation pour cette étape."
)


def main() -> None:
    payload = _load_stdin()
    if _should_continue(payload):
        out = {"followup_message": FOLLOWUP}
    else:
        out = {}
    sys.stdout.write(json.dumps(out, ensure_ascii=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
