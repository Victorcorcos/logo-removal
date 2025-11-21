#!/bin/bash
# Example usage script for logo removal tool
# This script demonstrates various ways to use the logo removal tool

echo "======================================"
echo "Logo Removal Tool - Example Usage"
echo "======================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Example 1: Basic usage
echo "Example 1: Basic Usage"
echo "----------------------"
echo "Process all images in a folder:"
echo ""
echo "  python remove_logos.py -i /path/to/images"
echo ""

# Example 2: Custom output directory
echo "Example 2: Custom Output Directory"
echo "-----------------------------------"
echo "Specify where to save processed images:"
echo ""
echo "  python remove_logos.py -i ~/Pictures/anime -o ~/Pictures/cleaned"
echo ""

# Example 3: Adjust detection sensitivity
echo "Example 3: Adjust Detection Sensitivity"
echo "----------------------------------------"
echo "Lower confidence for more detections (may include false positives):"
echo ""
echo "  python remove_logos.py -i /path/to/images -c 0.15"
echo ""
echo "Higher confidence for stricter detection:"
echo ""
echo "  python remove_logos.py -i /path/to/images -c 0.5"
echo ""

# Example 4: Increase mask expansion
echo "Example 4: Increase Mask Expansion"
echo "-----------------------------------"
echo "If logo edges are still visible, expand the mask:"
echo ""
echo "  python remove_logos.py -i /path/to/images -e 25"
echo ""

# Example 5: Force CPU usage
echo "Example 5: Force CPU Usage"
echo "----------------------------"
echo "If you don't have a GPU or want to save VRAM:"
echo ""
echo "  python remove_logos.py -i /path/to/images -d cpu"
echo ""

# Example 6: Verbose mode
echo "Example 6: Verbose Logging"
echo "---------------------------"
echo "See detailed processing information:"
echo ""
echo "  python remove_logos.py -i /path/to/images -v"
echo ""

# Example 7: Combined options
echo "Example 7: Combined Options"
echo "----------------------------"
echo "Use multiple options together:"
echo ""
echo "  python remove_logos.py -i ~/anime -o ~/cleaned -c 0.3 -e 20 -v"
echo ""

# Interactive example
echo "======================================"
echo "Would you like to run a test?"
echo "======================================"
echo ""
read -p "Enter path to test images (or press Enter to skip): " test_path

if [[ -n "$test_path" ]]; then
    if [[ -d "$test_path" ]]; then
        echo ""
        echo "Running logo removal on: $test_path"
        echo ""
        python remove_logos.py -i "$test_path" -v
        echo ""
        echo "✓ Processing complete!"
        echo "Check the results in: $test_path/no_logos/"
    else
        echo "⚠️  Directory not found: $test_path"
    fi
else
    echo "Skipping test. Use the examples above to get started!"
fi

echo ""
echo "======================================"
echo "For more information, see README.md"
echo "======================================"
