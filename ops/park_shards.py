#!/usr/bin/env python3
"""Detached companion to s03: park each decoder-layer shard on the Seagate once s03's `layer/L<L>`
unit is done (ops/lib/park.py has the why). Exits when all 40 are parked or s03 is complete.
    setsid nohup .venv/bin/python ops/park_shards.py >> ops/logs/park.log 2>&1 &"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.lib import env, park  # noqa: E402

while True:
    led = env.Ledger(env.LOGS / "s03_ledger.json")
    n = 0
    for L in range(40):
        if led.done(f"layer/L{L}"):
            n += park.park(L)
    if n == 40 or led.done("head"):
        env.log(f"park_shards exiting: {n}/40 layer shards parked", "park")
        break
    time.sleep(60)
