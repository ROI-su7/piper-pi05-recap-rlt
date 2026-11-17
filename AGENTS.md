# Project scope

Three-camera Piper pi0.5 collection/conversion/training/inference/takeover followed by either RECAP or RLT only. Keep recap/ and rlt/ implementations and training outputs separate. No world models, RISE or unrelated policy training.

# Source discipline

Source overlays/patches belong in common/, recap/, rlt/; never edit generated deps/ as the authoritative change. After intentional overlay changes run scripts/finalize_sources.py and reconstruct fresh dependencies. Retain original source licenses and hashes. Do not upload weights, datasets, credentials, logs or compiled workspaces.

# Takeover invariants

One SDK owner per physical CAN and one mux owner of follower command topics. Policies publish only to policy topics. Preserve powered HOLD, fresh executor ACK/TX-drain gates, confirmed takeover ACTIVE, actual-command recording and explicit fresh replan after ownership changes. Never replace measured state with a policy command. Keep original RTC timing and unit conventions.

# Verification

Run scripts/check.sh in the model environment after bootstrap. Offline tests must not construct hardware SDK objects or SocketCAN. New takeover behavior needs race/boundary tests and docs. Distinguish CPU/source reconstruction evidence from GPU training and actual robot execution.
