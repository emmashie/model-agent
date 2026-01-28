#!/bin/bash
# Quick Start Script for ROMS Grid Agent

echo "=========================================="
echo "ROMS Grid Agent - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Check if anthropic is installed
echo "Checking dependencies..."
python -c "import anthropic; print('✓ anthropic installed')" 2>/dev/null || echo "⚠ anthropic not installed (optional)"
python -c "import xarray; print('✓ xarray installed')" 2>/dev/null || echo "✗ xarray not installed (required)"
python -c "import numpy; print('✓ numpy installed')" 2>/dev/null || echo "✗ numpy not installed (required)"
python -c "import scipy; print('✓ scipy installed')" 2>/dev/null || echo "✗ scipy not installed (required)"
echo ""

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠ ANTHROPIC_API_KEY not set (LLM features will be disabled)"
    echo "  To enable: export ANTHROPIC_API_KEY='your-key-here'"
else
    echo "✓ ANTHROPIC_API_KEY is set"
fi
echo ""

echo "----------------------------------------"
echo "Available Commands:"
echo "----------------------------------------"
echo ""
echo "1. Run tests (no API key needed):"
echo "   python test_agent.py"
echo ""
echo "2. See usage examples:"
echo "   python examples.py"
echo ""
echo "3. Use the agent in your code:"
echo "   from llm_grid_agent import ROMSGridAgent"
echo "   agent = ROMSGridAgent(model_tools_path='...')"
echo "   result = agent.execute_workflow('Create a grid for Chesapeake Bay')"
echo ""
echo "4. Install optional dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "=========================================="
echo "Ready to go! 🚀"
echo "=========================================="
