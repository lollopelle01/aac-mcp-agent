#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"

osascript <<EOF
tell application "Terminal"
    activate

    -- Tab 1: backend
    set w to do script "venv_activate BDATM && cd '${ROOT}/app' && uvicorn src.api.server:app --reload --port 8000"

    -- Tab 2: frontend
    tell application "System Events" to keystroke "t" using command down
    delay 1.5
    do script "venv_activate BDATM && cd '${ROOT}/app/frontend' && npm run dev" in front window
end tell
EOF
