#!/usr/bin/env bash
# =============================================================================
# h200_probe.sh — 03_plan_h200 §R0：H200 硬體實測報告（比照平台 B 的 REPORT_MI300X.md）
#
# 由平台 B 的 lab_bin/05_hwinfo.sh（盤點）、25_storage_bench.sh（fio）、gpu_smoke.py、99_report.sh（報告）
# 改成 NVIDIA／Slurm 版，合成一支。
#
#   在 job 裡跑（要 GPU）：
#     sbatch --job-name=r0-h200-probe --time=01:00:00 code/tiara.sbatch bash code/h200_probe.sh
#   或 srun ... --pty bash 之後：source code/env_h200.sh; bash code/h200_probe.sh
#
# 產出：
#   $RUN_DIR/probe/            原始明細（純文字，可 diff；不進 git）
#   results/hw_h200.json       摘要（進 git）
#   docs/h200/REPORT_H200.md   報告（腳本產生、不要手改；每一節標來源檔）
# 原則（CLAUDE.md §1）：沒量到的寫 NOT_MEASURED，不寫估計值；失敗的照錄錯誤，不略過。
# GPU smoke 的數字**不可寫進論文**（數量級與「算得對」而已）。
#
# 可調（給測試用）：PROBE_FIO_SIZE（預設 4G）、PROBE_SKIP_FIO=1、PROBE_SKIP_GPU=1、
#                   PROBE_FIO_TARGETS="name=dir ..."（蓋掉預設的 home／shm／tmp）、HW_JSON、REPORT_MD
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
if [ "${TIARA_PLATFORM:-}" != C_H200 ]; then
  # shellcheck disable=SC1091
  source "$HERE/env_h200.sh"
fi
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="${RUN_DIR:-$TIARA_RUNS/$STAMP-h200-probe}/probe"
mkdir -p "$OUT"
HW_JSON="${HW_JSON:-$REPO/results/hw_h200.json}"
REPORT_MD="${REPORT_MD:-$REPO/docs/h200/REPORT_H200.md}"
echo "=== h200_probe.sh @ $(date -Is) -> $OUT ==="

# 可選指令：工具不在就記一行，不要讓整支掛掉（平台 B 05_hwinfo.sh 的教訓）
opt() { if command -v "$1" >/dev/null 2>&1; then "$@"; else echo "(缺少指令: $1)"; return 127; fi; }
cap() { local f="$OUT/$1"; shift; { "$@"; } > "$f" 2>&1; echo "exit=$?" >> "$f.exit"; }

# ---- 1. GPU --------------------------------------------------------------------
GPU_FIELDS="index,name,uuid,pci.bus_id,memory.total,driver_version,compute_cap,vbios_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit"
cap nvidia-smi.txt            opt nvidia-smi
cap nvidia-smi-L.txt          opt nvidia-smi -L
cap nvidia-smi-q.txt          opt nvidia-smi -q
cap nvidia-smi-topo.txt       opt nvidia-smi topo -m
cap nvidia-smi-nvlink.txt     opt nvidia-smi nvlink -s
cap nvidia-smi-apps.txt       opt nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
# 欄位逐個查：不支援的欄位會讓整個 --query-gpu 失敗
for f in ${GPU_FIELDS//,/ }; do
  printf '%s\t%s\n' "$f" "$(nvidia-smi --query-gpu="$f" --format=csv,noheader 2>&1 | head -1)"
done > "$OUT/gpu_fields.tsv" 2>&1

# ---- 2. CPU／NUMA／配額 ----------------------------------------------------------
cap lscpu.txt                 opt lscpu
cap numactl.txt               opt numactl -H
cap meminfo.txt               cat /proc/meminfo
cap self_cgroup.txt           cat /proc/self/cgroup
cap affinity.txt              python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print(len(a)); print(a)"
cap shm.txt                   df -B1 --output=source,fstype,size,avail,target /dev/shm
# cgroup：Slurm 的 job 在子 cgroup 裡。由自己的 cgroup 往上找第一個有上限的 memory.max（v2）／memory.limit_in_bytes（v1）
python3 - > "$OUT/cgroup_limits.txt" 2>&1 <<'PYCG'
import os
lines = open("/proc/self/cgroup").read().splitlines()
def walk(root, rel, names):
    parts = [p for p in rel.strip("/").split("/") if p]
    for k in range(len(parts), -1, -1):
        d = os.path.join(root, *parts[:k])
        for n in names:
            f = os.path.join(d, n)
            if os.path.exists(f):
                v = open(f).read().strip()
                if v not in ("max", "9223372036854771712"):
                    return f, v
    return None, None
for l in lines:
    hid, ctrls, rel = l.split(":", 2)
    if hid == "0":                       # v2
        f, v = walk("/sys/fs/cgroup", rel, ["memory.max"]); print(f"memory_max\t{v}\t{f}")
        f, v = walk("/sys/fs/cgroup", rel, ["cpu.max"]); print(f"cpu_max\t{v}\t{f}")
        f, v = walk("/sys/fs/cgroup", rel, ["cpuset.cpus.effective"]); print(f"cpuset\t{v}\t{f}")
    elif "memory" in ctrls.split(","):   # v1
        f, v = walk("/sys/fs/cgroup/memory", rel, ["memory.limit_in_bytes"]); print(f"memory_max\t{v}\t{f}")
PYCG

# ---- 3. Slurm 可見性（決定 gpu_guard 回 QUIET 還是 UNOBSERVABLE_HOST；03 §R0、H200_SLURM §5.3）-----------
NODE="${SLURMD_NODENAME:-$(hostname -s)}"
cap slurm_env.txt             bash -c 'env | grep -E "^SLURM_" | sort; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"'
cap scontrol_node.txt         opt scontrol show node "$NODE"
cap scontrol_job.txt          bash -c '[ -n "${SLURM_JOB_ID:-}" ] && scontrol show job "$SLURM_JOB_ID" || echo NOT_IN_SLURM'
cap squeue_node.txt           opt squeue -w "$NODE" -h -o '%i %u %a %b %C %m %T %M'
cap squeue_all.txt            opt squeue -h -o '%i %u %a %b %T %N'
cap sinfo.txt                 opt sinfo -N -l
cap gpu_guard_host.json       python "$REPO/code/gpu_guard.py" --host

# ---- 4. 軟體 -------------------------------------------------------------------
cap os.txt                    bash -c 'cat /etc/os-release; uname -a'
cap python_stack.txt          python -c "
import sys, importlib
print('python', sys.version.split()[0], sys.executable)
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
for m in ('vllm', 'transformers', 'flash_attn', 'triton', 'numpy', 'pandas'):
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, 'NOT_INSTALLED', type(e).__name__)
"
cap pip_freeze.txt            python -m pip freeze
cap conda_list.txt            opt conda list
cap repo.txt                  bash -c "git -C '$REPO' rev-parse HEAD; git -C '$REPO' status --porcelain | head -20"
cap network.txt               bash -c 'for u in https://huggingface.co https://pypi.org https://github.com; do printf "%s\t%s\n" "$u" "$(timeout 15 curl -sI "$u" 2>&1 | head -1 | tr -d "\r" || true)"; done'

# ---- 5. 儲存 -------------------------------------------------------------------
cap mounts.txt                bash -c 'df -hT; echo; findmnt -lo TARGET,SOURCE,FSTYPE,OPTIONS'
# M7_DISK 建議：$TIARA_RUNS 所在的區塊裝置（/proc/diskstats 的名字）。網路檔案系統沒有區塊裝置 → 改記本機根碟並註明
{
  src=$(findmnt -no SOURCE --target "$TIARA_RUNS" 2>/dev/null | head -1)
  fst=$(findmnt -no FSTYPE --target "$TIARA_RUNS" 2>/dev/null | head -1)
  echo -e "tiara_runs_source\t$src"; echo -e "tiara_runs_fstype\t$fst"
  dev=""; note=""
  if [[ "$src" == /dev/* ]]; then
    dev=$(basename "$(readlink -f "$src")"); note="TIARA_RUNS 所在的區塊裝置"
  else
    rsrc=$(findmnt -no SOURCE --target / 2>/dev/null | head -1)
    if [[ "$rsrc" == /dev/* ]]; then
      dev=$(basename "$(readlink -f "$rsrc")")
      note="TIARA_RUNS 在 $fst（非區塊裝置），diskstats 看不到；改記本機根碟 $rsrc。Guard 的磁碟欄這週不進判定（限速層，03 §5.1）"
    else
      note="TIARA_RUNS 在 $fst、根目錄也是 $rsrc，都不是區塊裝置 → 沒有可用的 M7_DISK（Guard 會丟錯）：照錄、回報，再決定 Guard 的磁碟欄怎麼處理"
    fi
  fi
  if [ -n "$dev" ] && awk -v d="$dev" '$3==d{f=1} END{exit !f}' /proc/diskstats; then ok=yes; else ok=no; fi
  echo -e "suggested_M7_DISK\t${dev:-NONE}"; echo -e "in_diskstats\t$ok"; echo -e "note\t$note"
  echo; awk '{print $3}' /proc/diskstats | tr '\n' ' '; echo
} > "$OUT/m7_disk.tsv" 2>&1

FIO_CSV="$OUT/storage_bench.csv"
echo "target,path,fstype,direct_ok,pattern,bs,bw_MBps,iops,lat_us_mean,lat_us_p99,note" > "$FIO_CSV"
if [ "${PROBE_SKIP_FIO:-0}" = 1 ]; then
  echo "SKIPPED (PROBE_SKIP_FIO=1)" > "$OUT/fio_skipped.txt"
elif ! command -v fio >/dev/null 2>&1; then
  echo "fio 不在（沒有 sudo，不自己裝）→ 儲存 NOT_MEASURED" > "$OUT/fio_skipped.txt"
else
  SIZE="${PROBE_FIO_SIZE:-4G}"
  fio --version > "$OUT/fio_version.txt" 2>&1
  if [ -n "${PROBE_FIO_TARGETS:-}" ]; then TARGETS="$PROBE_FIO_TARGETS"
  else TARGETS="home=$TIARA_HOME/.fio_bench shm=/dev/shm/.fio_bench_${USER:-u} tmp=${TMPDIR:-/tmp}/.fio_bench_${USER:-u}"; fi
  for spec in $TARGETS; do
    name=${spec%%=*}; DIR=${spec#*=}
    mkdir -p "$DIR" 2>/dev/null || { echo "$name,$DIR,,,ALL,,FAILED,,,,mkdir failed" >> "$FIO_CSV"; continue; }
    FSTYPE=$(df -T "$DIR" | awk 'NR==2{print $2}')
    need=$(numfmt --from=iec "$SIZE"); avail=$(df -B1 --output=avail "$DIR" | tail -1)
    if [ "$avail" -lt $((need * 2)) ]; then
      echo "$name,$DIR,$FSTYPE,,ALL,,NOT_MEASURED,,,,free $avail < 2x $SIZE" >> "$FIO_CSV"; rmdir "$DIR" 2>/dev/null; continue
    fi
    # O_DIRECT 探測：不支援時 --direct=1 會被靜默忽略，讀取數字被 page cache 灌水（平台 B 的教訓）
    DIRECT_OK=$(python3 -c "
import os, sys
p = os.path.join(sys.argv[1], '.odirect_probe')
try:
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600); os.close(fd); os.unlink(p); print('yes')
except OSError as e:
    print(f'no({e.errno})')
" "$DIR")
    for spec2 in "seqwrite:write:1m" "seqread:read:1m" "randwrite:randwrite:4k" "randread:randread:4k"; do
      IFS=: read -r label rw bs <<< "$spec2"
      J="$OUT/fio-$name-$label.json"
      if [[ "$label" == *read* ]] && [ -f "$DIR/testfile" ]; then   # 讀之前把檔案踢出 page cache
        python3 -c "import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" "$DIR/testfile"; sync
      fi
      if ! fio --name="$label" --filename="$DIR/testfile" --size="$SIZE" --rw="$rw" --bs="$bs" --direct=1 \
             --end_fsync=1 --ioengine=libaio --iodepth=16 --numjobs=1 --runtime=30 --time_based=0 \
             --group_reporting --output-format=json --output="$J" > "$OUT/fio-$name-$label.err" 2>&1; then
        echo "$name,$DIR,$FSTYPE,$DIRECT_OK,$label,$bs,FAILED,FAILED,FAILED,FAILED,see fio-$name-$label.err" >> "$FIO_CSV"; continue
      fi
      python3 - "$J" "$FIO_CSV" "$name" "$DIR" "$FSTYPE" "$DIRECT_OK" "$label" "$bs" <<'PYP'
import json, sys
j, csv, name, path, fstype, direct_ok, label, bs = sys.argv[1:9]
d = json.load(open(j))["jobs"][0]
sec = d["write"] if "write" in label else d["read"]
mean = p99 = float("nan")
for key, scale in (("clat_ns", 1000.0), ("lat_ns", 1000.0), ("clat", 1.0)):
    blk = sec.get(key)
    if not isinstance(blk, dict):
        continue
    pct = blk.get("percentile", {})
    for pk in ("99.000000", "99.00", "99"):
        if pk in pct:
            p99 = pct[pk] / scale
            break
    if blk.get("mean"):
        mean = blk["mean"] / scale
    if p99 == p99:
        break
open(csv, "a").write(f"{name},{path},{fstype},{direct_ok},{label},{bs},{sec['bw_bytes']/1e6:.1f},{sec['iops']:.0f},{mean:.1f},{p99:.1f},\n")
PYP
    done
    rm -f "$DIR/testfile"; rmdir "$DIR" 2>/dev/null
  done
fi

# ---- 6. GPU smoke --------------------------------------------------------------
if [ "${PROBE_SKIP_GPU:-0}" = 1 ]; then echo "SKIPPED (PROBE_SKIP_GPU=1)" > "$OUT/gpu_smoke.txt"
else cap gpu_smoke.txt python "$REPO/code/gpu_smoke.py" --json "$OUT/gpu_smoke.json"; fi

# ---- 7. 摘要 JSON ＋ 報告 ---------------------------------------------------------
mkdir -p "$(dirname "$HW_JSON")" "$(dirname "$REPORT_MD")"
python3 - "$OUT" "$HW_JSON" "$REPORT_MD" "${RUN_DIR:-}" <<'PYREP'
import csv, json, os, re, socket, sys
from datetime import datetime
OUT, HW_JSON, REPORT_MD, RUN_DIR = sys.argv[1:5]
NM = "NOT_MEASURED"

def rd(name):
    p = os.path.join(OUT, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""

def failed(name):
    t = rd(name + ".exit")
    return (not t) or ("exit=0" not in t)

def src(name):
    return f"`{os.path.join(OUT, name)}`"

hw = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "platform": "C_H200_Slurm",
      "hostname": socket.gethostname(), "run_dir": RUN_DIR or NM, "probe_dir": OUT,
      "_note": "h200_probe.sh 產生；NOT_MEASURED＝沒量到，不是 0。GPU smoke 不可寫進論文。"}

# --- 硬體
fields = {}
for line in rd("gpu_fields.tsv").splitlines():
    k, _, v = line.partition("\t")
    bad = (not v) or ("not supported" in v.lower() or "invalid" in v.lower() or "not found" in v.lower()
                      or "failed" in v.lower() or "N/A" in v)
    fields[k] = NM if bad else v.strip()
m = re.search(r"CUDA Version:\s*([\d.]+)", rd("nvidia-smi.txt"))
gpu = {k.replace(".", "_"): fields.get(k, NM) for k in
       ("name", "memory.total", "driver_version", "compute_cap", "pci.bus_id", "uuid", "vbios_version",
        "pcie.link.gen.max", "pcie.link.width.max", "pcie.link.gen.current", "pcie.link.width.current", "power.limit")}
gpu["cuda_driver_version"] = m.group(1) if m else NM
gpu["nvidia_smi_L_count"] = len([l for l in rd("nvidia-smi-L.txt").splitlines() if l.startswith("GPU ")]) \
    if not failed("nvidia-smi-L.txt") else NM
topo = rd("nvidia-smi-topo.txt").splitlines()
gpu["numa_affinity"] = gpu["cpu_affinity"] = NM
hdr = next((l for l in topo if "Affinity" in l), None)
row = next((l for l in topo if l.startswith("GPU0")), None)
if hdr and row:
    H = [h.strip() for h in hdr.split("\t")]
    R = [r.strip() for r in row.split("\t")]
    if len(R) == len(H):
        mp = dict(zip(H, R))
        gpu["numa_affinity"] = mp.get("NUMA Affinity", NM)
        gpu["cpu_affinity"] = mp.get("CPU Affinity", NM)
gpu["nvlink_status"] = NM if failed("nvidia-smi-nvlink.txt") else \
    (" | ".join(l.strip() for l in rd("nvidia-smi-nvlink.txt").splitlines()[:4]) or NM)
hw["gpu"] = gpu

def lscpu(key):
    for l in rd("lscpu.txt").splitlines():
        if l.split(":")[0].strip() == key:
            return l.split(":", 1)[1].strip()
    return NM
hw["cpu"] = {"model": lscpu("Model name"), "sockets": lscpu("Socket(s)"), "numa_nodes": lscpu("NUMA node(s)"),
             "host_nproc": lscpu("CPU(s)"),
             "numactl_nodes": " | ".join(l for l in rd("numactl.txt").splitlines() if "cpus:" in l)[:400] or NM}

# --- 配額
cg = {}
for l in rd("cgroup_limits.txt").splitlines():
    p = l.split("\t")
    if len(p) >= 3:
        cg[p[0]] = (p[1], p[2])
mem = cg.get("memory_max", ("None", ""))[0]
aff = rd("affinity.txt").splitlines()
shm = rd("shm.txt").splitlines()
memtotal = re.search(r"MemTotal:\s+(\d+)", rd("meminfo.txt"))
hw["quota"] = {
    "cpu_affinity_count": int(aff[0]) if aff and aff[0].isdigit() else NM,
    "cgroup_memory_max_gib": round(int(mem) / 2**30, 1) if mem.isdigit() else NM,
    "cgroup_memory_source": cg.get("memory_max", (None, NM))[1] or NM,
    "cgroup_cpuset": cg.get("cpuset", (NM, ""))[0] if cg.get("cpuset", ("None",))[0] != "None" else NM,
    "shm_size_gib": round(int(shm[-1].split()[2]) / 2**30, 1) if len(shm) >= 2 and shm[-1].split()[2].isdigit() else NM,
    "host_memtotal_gib": round(int(memtotal.group(1)) / 2**20, 1) if memtotal else NM,
}

# --- Slurm 可見性
node_txt = rd("scontrol_node.txt")
mg = re.search(r"\bGres=(\S+)", node_txt)
node_gpus = sum(int(x) for x in re.findall(r"gpu(?::[^:,(]+)?:(\d+)", mg.group(1))) if mg else None
try:
    guard = json.loads(rd("gpu_guard_host.json"))
except Exception:
    guard = {}
me = os.environ.get("USER", "")
users_all = {l.split()[1] for l in rd("squeue_all.txt").splitlines() if len(l.split()) > 1}
hw["slurm_visibility"] = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", NM), "node": os.environ.get("SLURMD_NODENAME", NM),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", NM),
    "nvidia_smi_L_count": gpu["nvidia_smi_L_count"], "node_gpus_from_scontrol": node_gpus if node_gpus else NM,
    "gpu_guard_host_level": guard.get("level", NM),
    "gpu_guard_error": NM if guard else rd("gpu_guard_host.json").strip()[-300:],
    "squeue_sees_other_users": (len(users_all - {me}) > 0) if not failed("squeue_all.txt") else NM,
    "cotenant_jobs_on_node": len([l for l in rd("squeue_node.txt").splitlines() if l.strip()])
                             if not failed("squeue_node.txt") else NM,
}

# --- 軟體
osr = re.search(r'PRETTY_NAME="([^"]+)"', rd("os.txt"))
stack = {}
for l in rd("python_stack.txt").splitlines():
    p = l.split()
    if p:
        stack[p[0]] = " ".join(p[1:])
hw["software"] = {"os": osr.group(1) if osr else NM, "kernel": (rd("os.txt").splitlines() or [NM])[-1][:200],
                  **{k: stack.get(k, NM) for k in ("python", "torch", "vllm", "transformers", "flash_attn", "triton")},
                  "repo_head": (rd("repo.txt").splitlines() or [NM])[0],
                  "pip_freeze_lines": len(rd("pip_freeze.txt").splitlines()) if not failed("pip_freeze.txt") else NM,
                  "network": {l.split("\t")[0]: (l.split("\t")[1] or "NO_RESPONSE") for l in rd("network.txt").splitlines()
                              if "\t" in l}}

# --- 儲存
rows = list(csv.DictReader(open(os.path.join(OUT, "storage_bench.csv")))) if os.path.exists(
    os.path.join(OUT, "storage_bench.csv")) else []
shm_seq = next((float(r["bw_MBps"]) for r in rows if r["target"] == "shm" and r["pattern"] == "seqread"
                and r["bw_MBps"].replace(".", "").isdigit()), None)
for r in rows:
    r["trust"] = "ok"
    if r["direct_ok"] != "yes":
        r["trust"] = "讀取不可信（O_DIRECT 不支援）" if "read" in r["pattern"] else "寫入僅供參考（無 O_DIRECT）"
    if (r["target"] != "shm" and r["pattern"] == "seqread" and shm_seq and r["bw_MBps"].replace(".", "").isdigit()
            and float(r["bw_MBps"]) >= shm_seq):
        r["trust"] = "不可信：比 tmpfs 還快 → 量到快取（判為 NOT_MEASURED）"
md = {}
for l in rd("m7_disk.tsv").splitlines():
    if "\t" in l:
        k, v = l.split("\t", 1)
        md[k] = v
hw["storage"] = {"fio": rows if rows else NM, "fio_note": rd("fio_skipped.txt").strip() or None,
                 "tiara_runs_source": md.get("tiara_runs_source", NM), "tiara_runs_fstype": md.get("tiara_runs_fstype", NM),
                 "suggested_M7_DISK": md.get("suggested_M7_DISK", NM), "M7_DISK_in_diskstats": md.get("in_diskstats", NM),
                 "M7_DISK_note": md.get("note", "")}

# --- GPU smoke
try:
    smoke = json.load(open(os.path.join(OUT, "gpu_smoke.json")))
except Exception:
    smoke = None
hw["gpu_smoke"] = smoke if smoke else {"status": NM, "detail": rd("gpu_smoke.txt").strip()[-400:]}

# --- 通過條件（03 §R0：GPU 是 H200、matmul 正確、可見性已判定）
name = gpu["name"] if gpu["name"] != NM else (smoke or {}).get("name", NM)
checks = {
    "gpu_is_h200": NM if name == NM else ("H200" in name),
    "matmul_correct": NM if not smoke else bool(smoke.get("correctness_ok")),
    "visibility_decided": hw["slurm_visibility"]["gpu_guard_host_level"] in ("QUIET", "LIGHT", "HEAVY", "UNOBSERVABLE_HOST"),
}
hw["r0_checks"] = checks
hw["r0_pass"] = all(v is True for v in checks.values())
json.dump(hw, open(HW_JSON, "w"), indent=2, ensure_ascii=False)
print(f"[ok] {HW_JSON}")

# ---------------------------------------------------------------- 報告
def v(x):
    return NM if x in (None, "", "None") else x
g, q, s, sw, st = hw["gpu"], hw["quota"], hw["slurm_visibility"], hw["software"], hw["storage"]
L = []
a = L.append
a("# 平台 C（NVIDIA H200 @ 學校 Slurm 叢集）硬體實測報告")
a("")
a(f"產生時間：{hw['ts']}　產生者：`code/h200_probe.sh`　節點：{hw['hostname']}　run：`{hw['run_dir']}`")
a("")
a("> **機器產生，不要手改。** 每一節標來源檔（原始明細在 run 目錄的 `probe/`，不進 git）。")
a("> `NOT_MEASURED`＝沒量到，**不是 0 也不是估計值**。比照平台 B 的 `REPORT_MI300X.md`（03_plan_h200 §R0）。")
a("")
a("## 0. §R0 通過條件")
a("")
a("| 條件 | 結果 |")
a("|:--|:--|")
for k, label in (("gpu_is_h200", "GPU 是 H200"), ("matmul_correct", "matmul 算得對"), ("visibility_decided", "Slurm 可見性已判定")):
    a(f"| {label} | {checks[k]} |")
a(f"| **整體** | **{'PASS' if hw['r0_pass'] else 'FAIL／未完成：照錄、停下來回報'}** |")
a("")
a(f"## 1. 硬體（來源：{src('gpu_fields.tsv')}、{src('nvidia-smi.txt')}、{src('nvidia-smi-topo.txt')}、{src('lscpu.txt')}）")
a("")
a("| 項目 | 值 |")
a("|:--|:--|")
for k, label in (("name", "GPU"), ("memory_total", "記憶體"), ("compute_cap", "compute capability"),
                 ("driver_version", "driver"), ("cuda_driver_version", "CUDA（driver 支援的版本）"),
                 ("pcie_link_gen_max", "PCIe 世代（max）"), ("pcie_link_width_max", "PCIe 寬度（max）"),
                 ("pcie_link_gen_current", "PCIe 世代（量的當下）"), ("pci_bus_id", "PCI bus"),
                 ("numa_affinity", "這張卡的 NUMA affinity"), ("cpu_affinity", "這張卡的 CPU affinity"),
                 ("nvidia_smi_L_count", "job 裡 nvidia-smi 看得到幾張卡"), ("nvlink_status", "NVLink（只記錄）")):
    a(f"| {label} | {v(g.get(k))} |")
a(f"| 節點上的卡數（scontrol Gres） | {v(s['node_gpus_from_scontrol'])} |")
a(f"| CPU | {v(hw['cpu']['model'])}（{v(hw['cpu']['sockets'])} socket、{v(hw['cpu']['numa_nodes'])} NUMA node）|")
a("")
a("> pinned 記憶體要配在 GPU 所在的 NUMA 節點（03 §4.1）：上表的 NUMA affinity。")
a("")
a(f"## 2. 配額（來源：{src('cgroup_limits.txt')}、{src('affinity.txt')}、{src('shm.txt')}）")
a("")
a("| 項目 | **job 真的能用的** | 機器上看到的（會騙人） |")
a("|:--|:--|:--|")
a(f"| CPU | **{v(q['cpu_affinity_count'])}**（sched_getaffinity；cpuset {v(q['cgroup_cpuset'])}） | lscpu CPU(s) = {v(hw['cpu']['host_nproc'])} |")
a(f"| RAM | **{v(q['cgroup_memory_max_gib'])} GiB**（{v(q['cgroup_memory_source'])}） | MemTotal = {v(q['host_memtotal_gib'])} GiB |")
a(f"| /dev/shm | {v(q['shm_size_gib'])} GiB | — |")
a("")
a(f"## 3. Slurm 可見性（來源：{src('gpu_guard_host.json')}、{src('nvidia-smi-L.txt')}、{src('scontrol_node.txt')}、{src('squeue_all.txt')}）")
a("")
a("| 項目 | 值 |")
a("|:--|:--|")
a(f"| SLURM_JOB_ID／節點／CUDA_VISIBLE_DEVICES | {v(s['slurm_job_id'])}／{v(s['node'])}／{v(s['cuda_visible_devices'])} |")
a(f"| job 看得到的卡／節點上的卡 | {v(s['nvidia_smi_L_count'])}／{v(s['node_gpus_from_scontrol'])} |")
a(f"| **gpu_guard.host_contention 的 level** | **{v(s['gpu_guard_host_level'])}** |")
a(f"| squeue 看得到別人的 job | {v(s['squeue_sees_other_users'])} |")
a(f"| 量的當下，同節點的 job 數（含自己） | {v(s['cotenant_jobs_on_node'])} |")
if s["gpu_guard_error"] != NM:
    a(f"| gpu_guard 錯誤 | `{s['gpu_guard_error']}` |")
a("")
a("> `UNOBSERVABLE_HOST`：同節點別人的負載看不到 → 時間數字的鄰居負載只能部分觀測（H200_SLURM §5.3），")
a("> 靠配置順序隨機打散＋配對比較＋每個 run 的 `squeue -w` 快照。`QUIET` 只在 job 看得到整個節點時才可能出現。")
a("")
a(f"## 4. 軟體（來源：{src('python_stack.txt')}、{src('os.txt')}、{src('pip_freeze.txt')}、{src('network.txt')}）")
a("")
a("| 項目 | 值 |")
a("|:--|:--|")
for k in ("os", "kernel", "python", "torch", "vllm", "transformers", "flash_attn", "triton", "repo_head", "pip_freeze_lines"):
    a(f"| {k} | {v(sw.get(k))} |")
for u, r in (sw.get("network") or {}).items():
    a(f"| 對外網路 {u} | {r} |")
a("")
a(f"## 5. 儲存（來源：{src('storage_bench.csv')}、{src('mounts.txt')}、{src('m7_disk.tsv')}）")
a("")
a("只記錄，這週不用真實裝置（03 §5.1：SSD／NFS 一律是限速層）。")
a("")
if isinstance(st["fio"], list) and st["fio"]:
    a("| 位置 | 路徑 | fstype | O_DIRECT | 型態 | MB/s | IOPS | 平均延遲(µs) | p99(µs) | 判讀 |")
    a("|:--|:--|:--|:--|:--|--:|--:|--:|--:|:--|")
    for r in st["fio"]:
        a(f"| {r['target']} | `{r['path']}` | {r['fstype']} | {r['direct_ok']} | {r['pattern']} | {r['bw_MBps']} | "
          f"{r['iops']} | {r['lat_us_mean']} | {r['lat_us_p99']} | {r['trust']}{'；' + r['note'] if r.get('note') else ''} |")
else:
    a(f"fio：{NM}（{st.get('fio_note') or '沒有結果'}）")
a("")
a(f"- `$TIARA_RUNS` 在：{v(st['tiara_runs_source'])}（{v(st['tiara_runs_fstype'])}）")
a(f"- **建議的 `M7_DISK`**：`{v(st['suggested_M7_DISK'])}`（在 /proc/diskstats：{v(st['M7_DISK_in_diskstats'])}）。{st['M7_DISK_note']}")
a("  → 填進 `code/env_h200.sh` 的 `M7_DISK` 預設值，或在 job 裡 `export M7_DISK=...`。")
a("- tmpfs（shm）那一列是對照組（純記憶體速度的上界）；任何位置的讀取 ≥ tmpfs，必然是量到快取。")
a("")
a(f"## 6. GPU smoke（來源：{src('gpu_smoke.json')}）")
a("")
sm = hw["gpu_smoke"]
if smoke:
    a("| 項目 | median | min | max |")
    a("|:--|--:|--:|--:|")
    for k, label in (("matmul_bf16_tflops", "BF16 matmul TFLOPS"), ("hbm_d2d_GBps", "HBM 裝置內複製 GB/s（讀＋寫）"),
                     ("h2d_pinned_GiBps", "pinned H2D GiB/s（1 GiB）"), ("d2h_pinned_GiBps", "pinned D2H GiB/s（1 GiB）")):
        x = sm.get(k) or {}
        a(f"| {label} | {x.get('median', NM)} | {x.get('min', NM)} | {x.get('max', NM)} |")
    a(f"| 正確性 max\\|I@Y−Y\\| | {sm.get('correctness_max_abs_err', NM)}（{'OK' if sm.get('correctness_ok') else 'FAIL'}） | | |")
    a("")
    a(f"torch {sm.get('torch')}、CUDA {sm.get('cuda')}、{sm.get('name')}、{sm.get('arch')}、torch 看得到 {sm.get('device_count')} 張")
else:
    a(f"{NM}：{sm.get('detail', '')}")
a("")
a("> ⚠️ smoke test，不是 benchmark：單一 stream、沒調 BLAS、每項 10 次。**不可寫進論文。**")
a("> 正式的 H2D／D2H 是 R2 的 C0（`code/m7_calib.py c0`）。限速層的三個頻寬都要明顯低於這裡的 pinned H2D（03 §4.1）。")
a("")
open(REPORT_MD, "w").write("\n".join(L) + "\n")
print(f"[ok] {REPORT_MD}")
print(json.dumps({"r0_pass": hw["r0_pass"], "r0_checks": checks, "gpu_guard_level": s["gpu_guard_host_level"],
                  "suggested_M7_DISK": st["suggested_M7_DISK"]}, ensure_ascii=False))
sys.exit(0 if hw["r0_pass"] else 1)
PYREP
rc=$?
echo "=== h200_probe.sh 完成（rc=$rc；0＝§R0 通過，1＝有條件沒過或沒量到：照錄、停下來回報）==="
exit $rc
