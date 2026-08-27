#!/bin/bash
cd "$(dirname "$0")"

pairs=(
  "qspec2024_complementary_quant:2410.11305"
  "synergy2023_batching_crossover:2310.18813"
  "specdec_meets_quant2025:2505.22179"
  "learning_to_draft2026_rl:2603.01639"
  "lossless_not_free2026_empirical:2607.17283"
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
