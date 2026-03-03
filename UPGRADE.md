# UPGRADE.md

Upgrade `fairseq2` 0.2→0.7 for Python 3.12 + numpy≥2.

## Blockers

- `fairseq2n==0.2` — no cp312 wheels (max cp311)
- `fairseq2==0.2` — requires `numpy<2`

## Scope

- `setup.py`: pin `fairseq2>=0.7`, drop `sonar-space`/`datasets` pins
- `fairseq2.assets.asset_store` singleton API changed in 0.7
- ~100 imports in `models/`, `streaming/agents/` use 0.2 APIs
- Key: `ConfigLoader`, `ModelLoader`, `ArchitectureRegistry`
- Upstream `fs2-update` branch (0.3) exists but skipped `streaming/`
