#!/bin/bash

# Script to convert all PDF files in the same folder to PNG,
# and copy existing PNG files to a subfolder named 'converted_png'.

# Strict mode: exit on error, undefined variable, or pipe failure
set -euo pipefail

# Determine the directory where this script resides and move there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working in directory: $SCRIPT_DIR"

# Create output folder if it does not exist
OUTPUT_DIR="converted_png"
mkdir -p "$OUTPUT_DIR"
echo "Output folder: $OUTPUT_DIR"

# ----------------------------------------------------------------------
# Helper: copy existing PNG files
# ----------------------------------------------------------------------
copy_png_files() {
    # Use nullglob to avoid literal '*.png' if no matches
    shopt -s nullglob
    for file in *.png; do
        echo "Copying existing PNG: $file"
        cp "$file" "$OUTPUT_DIR/"
    done
    shopt -u nullglob
}

# ----------------------------------------------------------------------
# Helper: convert PDF files using pdftoppm (preferred) or convert
# ----------------------------------------------------------------------
convert_pdf_files() {
    # Check which conversion tools are available
    if command -v pdftoppm >/dev/null 2>&1; then
        CONVERT_CMD="pdftoppm"
    elif command -v convert >/dev/null 2>&1; then
        CONVERT_CMD="convert"
    else
        echo "ERROR: Neither 'pdftoppm' (from poppler-utils) nor 'convert' (from ImageMagick) is installed."
        echo "Please install one of them, e.g.:"
        echo "  sudo apt install poppler-utils   # for pdftoppm"
        echo "  sudo apt install imagemagick     # for convert"
        exit 1
    fi

    echo "Using $CONVERT_CMD for PDF conversion."

    shopt -s nullglob
    for pdf in *.pdf; do
        echo "Converting PDF: $pdf"
        base="$(basename "$pdf" .pdf)"
        if [ "$CONVERT_CMD" = "pdftoppm" ]; then
            # pdftoppm: -png for PNG output, -r 150 for resolution, output prefix
            pdftoppm -png -r 150 "$pdf" "$OUTPUT_DIR/$base"
            # Note: pdftoppm always appends a page number, e.g. "name-1.png"
        else
            # convert (ImageMagick): -density 150 for quality, output filename
            # For multi-page PDFs, this creates name-0.png, name-1.png, etc.
            convert -density 150 "$pdf" "$OUTPUT_DIR/$base.png"
        fi
    done
    shopt -u nullglob
}

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
copy_png_files
convert_pdf_files

echo "Done. All PNG files are now in '$OUTPUT_DIR'."
