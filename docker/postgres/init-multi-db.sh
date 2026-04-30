#!/usr/bin/env bash
set -euo pipefail

echo "[init-multi-db] Ensuring deepfake_video DB exists..."
if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres -tAc "SELECT 1 FROM pg_database WHERE datname='deepfake_video'" | grep -q 1; then
  echo "[init-multi-db] deepfake_video already exists."
else
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres -c "CREATE DATABASE deepfake_video;"
  echo "[init-multi-db] deepfake_video created."
fi

