#!/usr/bin/env bash
set -euo pipefail

# Compile requirements.txt and create platform-specific variants
# Run this script from the project root.

echo "[compile_requirements.sh] Running pip-compile to generate requirements.txt"
if ! command -v pip-compile &>/dev/null; then
  echo "pip-compile not found in PATH; attempting to install pip-tools"
  pip install pip-tools
fi

if [ ! -f requirements.in ]; then
  echo "requirements.in not found in $(pwd)"
  exit 1
fi

# Generate requirements.txt
pip-compile --output-file=requirements.txt requirements.in

# Create Linux-specific file by copying and annotating
cp requirements.txt requirements-linux.txt
sed -i '' -e '1s;^;# Linux-specific requirements.txt\n# Platform: linux_x86_64\n# This is derived from requirements.txt\n;g' requirements-linux.txt 2>/dev/null || \
  sed -i '1s;^;# Linux-specific requirements.txt\n# Platform: linux_x86_64\n# This is derived from requirements.txt\n;g' requirements-linux.txt

# Create Windows-specific file by copying and annotating
cp requirements.txt requirements-windows.txt
sed -i '' -e '1s;^;# Windows-specific requirements.txt\n# Platform: win_amd64\n# This is derived from requirements.txt\n;g' requirements-windows.txt 2>/dev/null || \
  sed -i '1s;^;# Windows-specific requirements.txt\n# Platform: win_amd64\n# This is derived from requirements.txt\n;g' requirements-windows.txt

# Add short notes at end of each file
cat >> requirements-linux.txt <<'EOF'

# Linux-specific notes:
# - Install development headers and BLAS/LAPACK (e.g., libopenblas-dev)
# - Use system package manager for system libs when needed
EOF

cat >> requirements-windows.txt <<'EOF'

# Windows-specific notes:
# - Ensure Microsoft Visual C++ Build Tools installed
# - Prefer pre-built wheels for heavy numeric packages on Windows
EOF

echo "[compile_requirements.sh] Generated: requirements.txt, requirements-linux.txt, requirements-windows.txt"
