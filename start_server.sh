#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./audio.cpp/build/linux-cuda-release/bin/audiocpp_server --config ./server.json
