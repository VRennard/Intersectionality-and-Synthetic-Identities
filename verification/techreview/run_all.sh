#!/bin/bash
# Technical-review analysis batch (items 2,7,8b,9,10,11,12,16b,17,18b,20,21b)
cd "$(dirname "$0")"
PY=python3
for t in t1_rho_bootstrap t5_d3_endpoints_per_model t6_matched_triples \
         t4_base_steering t2_micro_batch t3_flagship_pass; do
  echo ""
  echo "================ $t $(date +%H:%M:%S) ================"
  $PY $t.py || echo "!!! $t FAILED"
done
echo ""
echo "================ BATCH DONE $(date +%H:%M:%S) ================"
