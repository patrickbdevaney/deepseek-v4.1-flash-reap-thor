#!/bin/bash
# QA front-load while s04 owns the GPU: CPU builds + CPU gates (marked through CHAIN_STEPS, so chain2 skips them later),
# then the two SMALL synthetic GPU suites (CUDA kernel units, llama.cpp test-backend-ops). Those load no model weights
# (a few GiB at most) and are run directly, NOT through CHAIN_STEPS: their reclaim() needs >=100 GiB free, which
# cannot hold while s04 is resident, and their done markers stay for the chain's own quiet-window run.
cd ~/deepseek-v4.1-flash-reap-thor; Q=ops/logs/qa; mkdir -p $Q
r() { local n=$1; shift; echo "[$(date -Is)] start $n" >> $Q/summary.log
  "$@" > $Q/$n.log 2>&1; local rc=$?; echo "[$(date -Is)] $n rc=$rc" >> $Q/summary.log; return $rc; }
floor() { a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ $a -ge 40 ] || { echo "[$(date -Is)] SKIP $1: MemAvailable $a GiB" >> $Q/summary.log; return 1; }; }
r llama_build  bash llamacpp/CHAIN_STEPS.sh build
r cuda_build   bash cuda/CHAIN_STEPS.sh build
r cuda_cpu_gates bash cuda/CHAIN_STEPS.sh cpu_gates
floor cuda_units && r cuda_units bash -c '[ -f cuda/build/units/moekt_ref.npy ] || .venv/bin/python -I cuda/tests/v41_unit_ref.py --out cuda/build/units; cuda/build/gate_v41_units cuda/build/units'
B=llamacpp/llama.cpp/build-thor/bin
floor backend_ops && r backend_ops_types bash -c "for t in mxfp4 q8_0 q6_k iq3_s iq3_xxs iq2_s iq2_xs iq2_xxs bf16 f32; do $B/test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS -p type_a=\$t || echo TYPE_FAIL \$t; done"
floor backend_ops_all && r backend_ops_all $B/test-backend-ops test -b CUDA0
echo "[$(date -Is)] qa_frontload finished" >> $Q/summary.log
