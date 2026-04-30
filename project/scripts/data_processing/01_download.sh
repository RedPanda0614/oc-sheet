#!/bin/bash
mkdir -p data/raw/expression_sheet
mkdir -p data/raw/character_sheet

# Download expression sheets (primary, for training)
gallery-dl \
  --range 1-3000 \
  --directory data/raw/expression_sheet \
  "https://safebooru.org/index.php?page=post&s=list&tags=expressions+multiple_views"

# Download multi-view character sheets (for future experiments)
# gallery-dl \
#   --range 1-3000 \
#   --directory data/raw/character_sheet \
#   "https://safebooru.org/index.php?page=post&s=list&tags=character_sheet+multiple_views"

echo "Download complete"
echo "expression_sheet: $(ls data/raw/expression_sheet/*.jpg 2>/dev/null | wc -l) images"
echo "character_sheet:  $(ls data/raw/character_sheet/*.jpg 2>/dev/null | wc -l) images"