#!/bin/bash
cd "$(dirname "$0")"

pairs=(
  "kim2025_gammatune:2504.00030"
  "speckv2026_adaptive_gamma:2605.02888"
  "adaedl2024_entropy_early_stop:2410.18351"
  "banditspec2025_bandit_gamma:2505.15141"
  "nightjar2025_dynamic_serving:2512.22420"
)

for pair in "${pairs[@]}"; do
  name="${pair%%:*}"
  id="${pair##*:}"
  url="https://arxiv.org/pdf/${id}"
  echo "Downloading $name ($id)..."
  curl -sL -A "Mozilla/5.0 (research download)" "$url" -o "${name}.pdf"
  size=$(stat -f%z "${name}.pdf" 2>/dev/null || echo 0)
  filetype=$(file -b "${name}.pdf")
  echo "  -> size=${size} bytes, type=${filetype}"
done
