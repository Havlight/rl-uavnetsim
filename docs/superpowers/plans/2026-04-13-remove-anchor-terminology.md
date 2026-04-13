# Remove Anchor Terminology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining `anchor` compatibility layer and rename the codebase to use only `gateway-capable` / `active gateway` terminology without regressing single-gateway satellite behavior.

**Architecture:** Eliminate deprecated `anchor` fields, parameters, helpers, and accounting aliases from production code first, then update tests and visualization call sites to use gateway-native names only. Keep the actual single-gateway satellite behavior unchanged while cleaning the public API and internal naming.

**Tech Stack:** Python, pytest, numpy, matplotlib dataclasses, existing `rl_uavnetsim` environment/network stack

---

### Task 1: Remove anchor-based test usage
- [ ] Update relay, RL, and visualization tests to construct gateway UAVs with `is_gateway_capable=True`
- [ ] Replace `anchor_uav_id` constructor and helper arguments with `gateway_capable_uav_ids` / `gateway_uav_ids`
- [ ] Replace widest-path helper coverage with routing-table coverage

### Task 2: Remove anchor compatibility from production APIs
- [ ] Drop `is_anchor` from `UAV`
- [ ] Drop `anchor_uav_id` from `SimEnv`, `execute_relay_service`, and visualization helpers
- [ ] Rename default config constant from `ANCHOR_UAV_ID` to `DEFAULT_GATEWAY_UAV_ID`

### Task 3: Remove anchor-specific data structures and exports
- [ ] Remove `RelayPath` / `find_widest_path_to_anchor`
- [ ] Remove `anchor_*` accounting aliases and `total_relay_in_bits_to_anchor`
- [ ] Clean up exports and remaining string/variable references

### Task 4: Verify and preserve behavior
- [ ] Run focused pytest targets during the red/green cycle
- [ ] Run full `~/.venv/bin/python -m pytest -q tests`
- [ ] Run satellite demo smoke validation after the rename cleanup
