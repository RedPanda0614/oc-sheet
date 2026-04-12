#!/bin/bash
mkdir -p data/raw/expression_sheet
mkdir -p data/raw/character_sheet

# 下载表情合集图（优先，训练用）
gallery-dl \
  --range 1-3000 \
  --directory data/raw/expression_sheet \
  "https://safebooru.org/index.php?page=post&s=list&tags=expressions+multiple_views"

# 下载多视角设定图（后续视角实验用）
# gallery-dl \
#   --range 1-3000 \
#   --directory data/raw/character_sheet \
#   "https://safebooru.org/index.php?page=post&s=list&tags=character_sheet+multiple_views"

echo "下载完成"
echo "expression_sheet: $(ls data/raw/expression_sheet/*.jpg 2>/dev/null | wc -l) 张"
echo "character_sheet:  $(ls data/raw/character_sheet/*.jpg 2>/dev/null | wc -l) 张"