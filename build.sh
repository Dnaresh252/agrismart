#!/usr/bin/env bash
set -o errexit

echo "🚀 Building AgriSmart AI Platform..."

pip install -r requirements.txt
mkdir -p staticfiles
python manage.py collectstatic --noinput --clear
python manage.py migrate --noinput

echo "✅ AgriSmart deployment ready!"