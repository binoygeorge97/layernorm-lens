# D6 planner data: provenance note, 3 October 2026

Facts read from the committed files of commit 3d52d3d (`results/r6/planner/`,
`results/r6/planner_smoke/`) and from git. Nothing here changes any rule or any
stored observation. Written before any criterion quantity was computed.

## 1. The interrupted session 20260925T214748Z

- It collected cartpole-swingup seed 2 and left no session file. Its files are
  `meta_cartpole-swingup-seed2.json` and `gate/gate_20260925T214748Z.json`.
- Both record `git_commit` bca1e71bf749642f274fcc7a990f22011560ade4 with
  `git_dirty` false.
- The main session, 20260927T141627Z, recorded the same commit, clean: its session
  file and all 14 other metas.
- Both sessions therefore ran the same code. `planner_collect.py` and
  `planner_lib.py` are identical between them, because the commit is the same.
- Both ran on a Tesla T4 with tdmpc2 e9f59321933cbc8e11a002b842adc7d4ffae8ff1.

## 2. Compiled and eager planning

- Every planner meta records `tdmpc2_config.compile` true.
- In tdmpc2, `torch.compile` wraps `_plan` (`tdmpc2/tdmpc2.py:50-51`:
  `plan = torch.compile(self._plan, mode="reduce-overhead")`). `encode` is not
  compiled.
- The encoder agreement gate calls `agent.model.encode` directly, so the gate
  tested `encode` eagerly in every case, as D6 (b) states.
- In the main session, torch._dynamo reached its recompile limit (8; a guard on
  `kwargs['t0']`). This was seen in the Colab session output. No committed file
  records the message, so the committed evidence is the per-episode times
  (`seconds_per_episode` in each meta).
- A compiled run shows a long first episode, which includes compilation, and then
  about 9.5 s per episode. That matches planner_check's compiled runs, which D6
  records as about 9 s for cartpole and 13 s for humanoid.
- An eager run shows no long first episode and 21–29 s per episode.

| Checkpoint | Session | First episode (s) | Median (s) | Planning |
| --- | --- | --- | --- | --- |
| cartpole-swingup s2 | 20260925T214748Z | 32.2 | 9.59 | compiled |
| cartpole-swingup s3 | 20260927T141627Z | 32.4 | 9.49 | compiled |
| cheetah-run s1 | 20260927T141627Z | 126.4 | 9.48 | compiled |
| cheetah-run s2 | 20260927T141627Z | 29.6 | 9.47 | compiled |
| cheetah-run s3 | 20260927T141627Z | 30.6 | 9.47 | compiled |
| walker-run s1 | 20260927T141627Z | 22.2 | 21.64 | eager |
| walker-run s2 | 20260927T141627Z | 22.3 | 21.07 | eager |
| walker-run s3 | 20260927T141627Z | 21.7 | 21.43 | eager |
| humanoid-run s1 | 20260927T141627Z | 28.0 | 27.60 | eager |
| humanoid-run s2 | 20260927T141627Z | 27.6 | 27.84 | eager |
| dog-run s1 | 20260927T141627Z | 29.6 | 29.26 | eager |
| dog-run s2 | 20260927T141627Z | 28.9 | 29.19 | eager |
| dog-run s3 | 20260927T141627Z | 29.4 | 29.39 | eager |
| cartpole-swingup s1 (pre-release) | 20260927T141627Z | 21.5 | 21.18 | eager |
| humanoid-run s3 (pre-release) | 20260927T141627Z | 27.1 | 26.73 | eager |

The rows are in collection order. Walker-run s1 has no compile-length first episode
(22.2 s), so it planned eagerly from its first episode on. Every later checkpoint in
the session did too. Cartpole-swingup s1 (eager, 21.18 s) against s2 and s3 of the
same task (compiled, 9.5 s) shows the difference on the same task.

No rule depends on this. It is disclosure against D6 (e)'s "`torch.compile` as in
tdmpc2's configuration": compilation was configured for all 15, and it took effect
for 5. The stored observations are the data of record (D6 (a)).

## 3. The two smoke sessions

| Session | Commit | Checkpoint | Episodes | Data file (kind `smoke`) |
| --- | --- | --- | --- | --- |
| 20260925T211407Z | 54a728881056514c4fd534d43745d9273ebe058d, clean | cartpole-swingup s1 | 1 | `data/r6/planner_smoke/cartpole-swingup-seed1-smoke-20260925T211407Z.npz` (fbaf61b4…) |
| 20260927T141335Z | bca1e71bf749642f274fcc7a990f22011560ade4, clean | cartpole-swingup s1 | 1 | `data/r6/planner_smoke/cartpole-swingup-seed1-smoke-20260927T141335Z.npz` (a81ee0b7…) |

- Both metas record kind `smoke` and the note "outcome data for this checkpoint;
  never used for any rule and never counted toward the 50 episodes".
- 54a7288 and bca1e71 differ only in `r6_planner_collect.ipynb` (`git diff --stat
  54a7288 bca1e71`).
- The criterion stage reads only kind `data` rows of `planner_obs_manifest.csv`. It
  never opens a smoke file.

## 4. The missing smoke key mapping (30 files, not 31)

- `results/r6/planner_smoke/key_mapping_humanoid-run-seed3.json` never reached
  Drive, so 3d52d3d has 30 files.
- The cause is in `planner_collect.py` at bca1e71. `load_agent` writes a pre-release
  checkpoint's key mapping when it loads it, and that happens for the gate. But
  `run_checkpoint` copies the mapping to Drive only for a checkpoint that acts
  (lines 472-473). In a smoke run only cartpole-swingup s1 acts.
- The mapping is a deterministic function of the checkpoint's key names, in order:
  `planner_lib.remap_keys(keys)`, written by `planner_collect.write_json` with the
  file name, checkpoint, SHA-256 and the D6 table rows. Evidence:
  - The smoke and full-run copies of cartpole-swingup s1's mapping
    (`planner_smoke/` and `planner/`) are byte-identical.
  - humanoid-run s3's mapping was regenerated on 3 Oct 2026 from
    `results/r6/prerelease_keys.json` (60 keys, `humanoid-run-3`). The same 60 key
    names, in the same order, were also read from the downloaded checkpoint, whose
    SHA-256 matches D6's table.
  - `planner_lib.py` is unchanged since bca1e71.
  - Serialised as `write_json` does (`json.dump(obj, f, indent=2)` into a text-mode
    file, which on Colab's Linux writes "\n"), the regenerated file is
    byte-identical to `results/r6/planner/key_mapping_humanoid-run-seed3.json`:
    SHA-256 b6af75c0393b2289de1ded75480391b63ac6f8e572d392c267705e8907dc4015, 3,666
    bytes, both.
  - Written by the same function on Windows, text mode gives CRLF line endings
    instead. The content is the same; only the line endings differ.
- The smoke run therefore wrote the same mapping as the `planner/` copy. Nothing is
  lost, and the 30 committed files are complete for every purpose.
- The copy gap itself is task (b) in `docs/STATE.md`.

Regeneration, as run (project venv, repository root):

```python
import json, hashlib, sys
sys.path.insert(0, "experiments/r6_tdmpc2")
import planner_lib as pl, provenance as pv, planner_collect as pc
keys = list(json.load(open("results/r6/prerelease_keys.json"))["humanoid-run-3"])
obj = dict(task="humanoid-run", seed=3, checkpoint=pv.checkpoint_name("humanoid-run", 3),
           sha256=pv.d6_sha_table()["dmcontrol/humanoid-run-3.pt"],
           d6_table=[row for row, _ in pl.PRERELEASE_REMAP], mapping=pl.remap_keys(keys),
           note="Q-ensemble keys pass through unchanged and are converted by "
                "tdmpc2's own api_model_conversion inside TDMPC2.load.")
b = json.dumps(obj, indent=2, default=pc._json_default).encode("utf-8")
ref = open("results/r6/planner/key_mapping_humanoid-run-seed3.json", "rb").read()
assert b == ref, "differs"
print(hashlib.sha256(b).hexdigest())
```

## Addendum, 4 October 2026: the session's console log

`results/r6/planner/logs/console_20260927T141627Z.txt` (SHA-256 d11f3774a1deacd8fd98d6fb3e2033681527bc782b0956061d1b81e81144a0a0) is the author's
transcription of the Colab cell-7 output of session 20260927T141627Z. It was made from
a copy pasted into the author's advisor chat; the runtime itself is gone. It covers the
14 checkpoints collected in that session. cartpole-swingup s2, from the earlier session
20260925T214748Z, is not in it.

It was verified against the data on 4 Oct 2026, with each planner observation file
first checked on Drive against its SHA-256 in `planner_obs_manifest.csv`:

- Every printed episode return (700 lines: 14 checkpoints × 50 episodes) equals the sum
  of that episode's stored rewards, matched by `env_seed` and `episode_in_seed`, at the
  printed 0.1.
- Every printed mean return, published return and fraction (14 lines) matches its
  kind `data` row of `planner_returns.csv`.
- Every printed episode time equals the meta's `seconds_per_episode` at the printed
  0.1 s.

There were no mismatches.

The recompile-limit warning is in the log, lines 257-261, at 14:51:36 UTC. It appears
after walker-run s1 was loaded and before that checkpoint's first episode line:
`torch._dynamo hit config.recompile_limit (8)`, function `inner`, last reason
`0/7: ___check_obj_id(kwargs['t0'], ...)`. This supersedes section 2's statement that
no committed file records the message. The compiled/eager split in section 2's table
stands, and the log now backs it alongside the per-episode times.
