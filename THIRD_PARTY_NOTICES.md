# Sources and attribution

This repository packages the user's Piper integration as scoped source overlays and patches. Upstream frameworks are reconstructed under ignored deps/ at exact commits, with SHA256 checks of each local change.

| Source | Revision / treatment |
|---|---|
| RLinf | c5ca51cc21c007a41d287159f9e1b14e0200000e; Apache-2.0; original headers retained; Piper RECAP/RLT modifications separated by directory |
| OpenPI | 15a9616a00943ada6c20a0f158e3adb39df2ccac; Apache-2.0; native JAX/runtime local changes retained; only three Piper pi0.5 task presets exposed |
| Piper ROS from user's cobot_magic checkout | a1c75e917c41780dd6575792f0e57636eff4295d plus local modifications; MIT notice retained in the workspace and licenses/AGILEX_MIT.txt |
| Cobot data collection / conversion | User-provided cobot_magic/collect_data sources; source comments retained; generated data/docs/media/build products omitted |
| Powered-HOLD follower mux | User's lingbot-vla-v2/deploy/piper_real/ros; Apache-2.0 package declaration; attributed as local Piper control adaptation; unrelated model code omitted |

Apache-2.0 in the root applies to the new integration and Apache-derived code; source-specific MIT notices continue to apply. Upstream model/tokenizer/SDK licenses remain with their publishers; pretrained weights and datasets are not redistributed.

Source changes relative to the inspected current workspace are portability adapters (environment paths, explicit source versions, entrypoints), scoped model presets, demonstration/rollout dataset composition, test fixture paths and offline transport guards. Robot driver/mux state machines are preserved. The CPU test SDK stub is only in pytest conftest and cannot be used by ROS launch.
