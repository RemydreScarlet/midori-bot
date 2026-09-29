#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
pkill -f "venv/bin/python bot.py" 2>/dev/null
sleep 1
setsid ./.venv/bin/python bot.py > /tmp/opencode/bot.log 2>&1 < /dev/null &
sleep 8
if pgrep -f "venv/bin/python bot.py" >/dev/null; then echo BOT_RUNNING; else echo BOT_FAILED; fi
tail -5 /tmp/opencode/bot.log
