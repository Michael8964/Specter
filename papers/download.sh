#!/bin/bash
cd "$(dirname "$0")"

pairs=(
  "leviathan2023_speculative_decoding:2211.17192"
  "chen2023_speculative_sampling_deepmind:2302.01318"
  "lin2023_awq:2306.00978"
  "frantar2023_gptq:2210.17323"
  "xiao2023_smoothquant:2211.10438"
  "kwon2023_vllm_pagedattention:2309.06180"
  "li2024_eagle:2401.15077"
  "li2024_eagle2:2406.16858"
  "cai2024_medusa:2401.10774"
  "liu2023_agentbench:2308.03688"
  "qin2023_toolllm:2307.16789"
  "survey2025_speculative_decoding:2502.19732"
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
