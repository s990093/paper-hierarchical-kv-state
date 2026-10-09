#!/bin/bash
# 08 消融的 GPU 確認（08_ablation_plan.md §5 第 4 步）。模擬器挑出的候選設定＋模擬器在新設定（doc、hold）上的驗證
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh
cd /mlsteam/workspace/paper-hierarchical-kv-state/code
export M7_IO_MODEL=share
S8="S1 S2b S4+ S4L S4B S5 S5L S5c"
step() { echo "=== $(date -Is) $1"; shift; "$@"; echo "exit=$?"; }
# G1：CPU 3.69（vLLM 實測）＋hold＋doc＋NFS；seed 0（模擬裡 S5L 贏 29%）與 seed 1（模擬裡 S1 贏）
M7_CPU_GIBPS=3.69 step G1-seed0 m7run m7-b2-g1-doc-hold-cpu3.69-s0 python m7_write_policy.py b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed 0 --strategies $S8 --cpu-fracs 0.25 0.5 --reps 3 --verify
M7_CPU_GIBPS=3.69 step G1-seed1 m7run m7-b2-g1-doc-hold-cpu3.69-s1 python m7_write_policy.py b --b2 --workload doc --release hold --ssd-dev nfs --wl-seed 1 --strategies $S8 --cpu-fracs 0.25 0.5 --reps 3 --verify
# G2：hold 單獨（chat、CPU 35.4、NFS、seed 0）：模擬說寫穿 S1 會大贏，驗證 hold 模型
step G2-chat-hold m7run m7-b2-g2-chat-hold-s0 python m7_write_policy.py b --b2 --workload chat --release hold --ssd-dev nfs --wl-seed 0 --strategies S1 S2b S4+ S4B S5 S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
echo "=== $(date -Is) DONE"
