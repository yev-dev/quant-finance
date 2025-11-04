#!/bin/bash

# Requirements compilation script for quant-finance project
# This script generates platform-specific requirements files using pip-compile

set -e  # Exit on any error

echo "🔧 Starting requirements compilation..."

# Check if pip-tools is installed
if ! command -v pip-compile &> /dev/null; then
    echo "❌ pip-tools not found. Installing..."
    pip install pip-tools
fi

# Check if requirements.in exists
if [ ! -f "requirements.in" ]; then
    echo "❌ requirements.in not found!"
    exit 1
fi

echo "📦 Compiling general requirements..."
pip-compile --output-file requirements.txt requirements.in

echo "🐧 Creating Linux-specific requirements..."
# Copy general requirements and add Linux-specific notes
cp requirements.txt requirements-linux.txt

# Add Linux-specific header
sed -i '1i#\n# Linux-specific requirements.txt\n# Platform: Linux (linux_x86_64)\n# Note: Some packages may have different versions or dependencies on Linux\n#' requirements-linux.txt

# Add Linux-specific notes at the end
cat >> requirements-linux.txt << 'EOF'

# Linux-specific notes:
# - Install development headers: sudo apt-get install python3-dev build-essential (Ubuntu/Debian)
# - Install BLAS/LAPACK: sudo apt-get install libopenblas-dev liblapack-dev
# - For CentOS/RHEL: sudo yum install python3-devel gcc blas-devel lapack-devel
# - Consider using system package manager for some dependencies like lxml: sudo apt-get install libxml2-dev libxslt1-dev
# - For better performance with numerical libraries, install Intel MKL or OpenBLAS
# - Some packages support parallel processing through OpenMP on Linux
EOF

echo "🪟 Creating Windows-specific requirements..."
# Copy general requirements and add Windows-specific notes
cp requirements.txt requirements-windows.txt

# Add Windows-specific header
sed -i '1i#\n# Windows-specific requirements.txt\n# Platform: Windows (win_amd64)\n# Note: Some packages may have different versions or dependencies on Windows\n#' requirements-windows.txt

# Add Windows-specific notes at the end
cat >> requirements-windows.txt << 'EOF'

# Windows-specific notes:
# - Ensure Microsoft Visual C++ 14.0 or greater is installed for packages with native extensions
# - Some packages like lxml, numpy, scipy might require pre-compiled wheels
# - For financial libraries, consider using Intel MKL optimized versions if available
# - PowerShell or Command Prompt can be used for installation
EOF

echo "✅ Requirements compilation completed!"
echo "📄 Generated files:"
echo "   - requirements.txt (general)"
echo "   - requirements-linux.txt (Linux-specific)"
echo "   - requirements-windows.txt (Windows-specific)"
echo ""
echo "💡 To install dependencies:"
echo "   Linux:   pip install -r requirements-linux.txt"
echo "   Windows: pip install -r requirements-windows.txt"
echo "   General: pip install -r requirements.txt"
