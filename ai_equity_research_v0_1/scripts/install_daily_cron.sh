#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Create the venv and install requirements first." >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
CRON_LINE="30 6 * * 1-6 cd '$ROOT' && '$PYTHON' -m src.daily_update >> '$ROOT/logs/cron.log' 2>&1"
( crontab -l 2>/dev/null | grep -Fv 'src.daily_update' || true; echo "$CRON_LINE" ) | crontab -
echo "Installed daily update at 06:30 in the machine's local timezone Monday-Saturday. Schedule after the latest market close relevant to your universe."
echo "Review: crontab -l"
echo "Logs: $ROOT/logs/daily_update.log and $ROOT/logs/cron.log"
