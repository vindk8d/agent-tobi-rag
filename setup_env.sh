#!/bin/bash

# Setup script for agent-tobi-rag development environment
# This script ensures proper Python path configuration for imports

echo "Setting up development environment..."

# Activate virtual environment
source venv/bin/activate

# Add current directory to PYTHONPATH for proper import resolution
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "✓ Virtual environment activated"
echo "✓ PYTHONPATH configured: $PYTHONPATH"
echo "✓ Python version: $(python --version)"

# Test critical imports
echo "Testing critical imports..."

python -c "
import sys
print('Python path includes:')
for p in sys.path[:5]:  # Show first 5 entries
    print(f'  - {p}')

# Test imports
test_imports = [
    ('pydantic', 'pydantic'),
    ('langchain_core.tools', 'langchain_core.tools'),
    ('langsmith', 'langsmith'),
    ('pytest', 'pytest'),
    ('backend.agents.memory', 'backend.agents.memory'),
    ('backend.agents.hitl', 'backend.agents.hitl'),
    ('backend.agents.toolbox.generate_quotation', 'backend.agents.toolbox.generate_quotation')
]

failed_imports = []
for module_name, import_path in test_imports:
    try:
        __import__(import_path)
        print(f'✓ {module_name}')
    except ImportError as e:
        print(f'✗ {module_name}: {e}')
        failed_imports.append(module_name)

if failed_imports:
    print(f'\n⚠️  {len(failed_imports)} import(s) failed')
    exit(1)
else:
    print(f'\n🎉 All imports successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Development environment ready!"
    echo "   Run tests with: python -m pytest tests/"
    echo "   Start backend with: python -m backend.main"
else
    echo ""
    echo "❌ Environment setup failed. Check error messages above."
    exit 1
fi




