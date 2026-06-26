"""
Phase 7 contract test — parallel council execution.

Unit tests (no Ollama required):
  - check_parallel_viable() returns without raising
  - threshold=0 always returns viable=True
  - threshold=999 always returns viable=False

Live test (--live, requires Ollama):
  - Both paths produce 4 RoleResponse objects in COUNCIL_ROLES order
  - Wall-clock comparison printed for reference

Usage:
    python test_parallel.py           # unit tests only
    python test_parallel.py --live    # requires Ollama running
"""

import argparse
import asyncio
import sys
import time

from core.runner import (
    COUNCIL_ROLES,
    check_parallel_viable,
    run_parallel,
    run_sequential,
)

HAMON = (
    "If Hamon were to prove that deep compatibility correlates only with a set of "
    "universally measurable behavioral metrics (e.g., the capacity for non-violent "
    "conflict resolution; resilience in ambiguity), would the convergence between "
    "MBTI and Vedic astrology become redundant? If yes, which part of your synthesis "
    "\u2014 the empirical rigor or the symbolic depth \u2014 is doing all the heavy "
    "lifting simply to maintain structural weight?"
)

_SEP = "=" * 64
OK  = "[PASS]"
ERR = "[FAIL]"


def test_ram_check() -> bool:
    """check_parallel_viable() returns a typed result without raising."""
    print("=== RAM check ===\n")
    viable, reason = check_parallel_viable()
    passed = isinstance(viable, bool) and isinstance(reason, str) and len(reason) > 0
    status = "viable" if viable else "not viable"
    print(f"{OK if passed else ERR}  check_parallel_viable() returned ({viable}, ...)")
    print(f"  status: {status}")
    print(f"  reason: {reason}\n")
    return passed


def test_ram_check_threshold_zero() -> bool:
    """threshold=0 always returns viable=True regardless of actual RAM."""
    print("=== RAM check — threshold=0 (always viable) ===\n")
    viable, reason = check_parallel_viable(threshold_gb=0.0)
    passed = viable is True
    print(f"{OK if passed else ERR}  threshold=0 -> viable=True")
    print(f"  viable: {viable}  reason: {reason}\n")
    return passed


def test_ram_check_threshold_huge() -> bool:
    """threshold=999 always returns viable=False (no machine has 999GB free)."""
    print("=== RAM check — threshold=999 (never viable) ===\n")
    viable, reason = check_parallel_viable(threshold_gb=999.0)
    passed = viable is False
    print(f"{OK if passed else ERR}  threshold=999 -> viable=False")
    print(f"  viable: {viable}  reason: {reason}\n")
    return passed


async def test_parallel_live() -> bool:
    """Both runners produce 4 responses in COUNCIL_ROLES order. Timing printed for reference."""
    print("=== Live: parallel vs sequential ===\n")

    print("Running sequential...")
    t0 = time.monotonic()
    seq_responses = await run_sequential(HAMON)
    seq_ms = int((time.monotonic() - t0) * 1000)

    print("Running parallel...")
    t0 = time.monotonic()
    par_responses = await run_parallel(HAMON)
    par_ms = int((time.monotonic() - t0) * 1000)

    expected_roles = [r.role for r in seq_responses]
    actual_roles   = [r.role for r in par_responses]
    roles_match    = actual_roles == expected_roles
    count_match    = len(par_responses) == len(COUNCIL_ROLES)
    passed         = roles_match and count_match

    print(f"\n{OK if passed else ERR}  output shape")
    print(f"  sequential: {seq_ms}ms  {[r.role.value for r in seq_responses]}")
    print(f"  parallel:   {par_ms}ms  {[r.role.value for r in par_responses]}")
    print(f"  roles_match: {roles_match}  count_match: {count_match}")

    if par_ms > 0:
        speedup = seq_ms / par_ms
        print(f"\n  wall-clock speedup: {speedup:.2f}x")
        print(f"  note: Virgo+Pisces share qwen3:8b — concurrent requests to one model")
        print(f"        cap throughput on that pair regardless of gather concurrency.")

    print(f"\nParallel responses:")
    for r in par_responses:
        outcome = "ok" if r.ok else f"error: {r.error}"
        print(f"  {r.role.value:<10} {r.model:<20} {r.duration_ms}ms  {outcome}")

    print()
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 parallel execution contract test")
    parser.add_argument("--live", action="store_true", help="Run live Ollama tests (requires ollama serve)")
    args = parser.parse_args()

    results = [
        test_ram_check(),
        test_ram_check_threshold_zero(),
        test_ram_check_threshold_huge(),
    ]

    if args.live:
        results.append(asyncio.run(test_parallel_live()))

    print(_SEP)
    all_ok = all(results)
    print("ALL PASS" if all_ok else "FAILURES DETECTED")
    sys.exit(0 if all_ok else 1)
