#!/usr/bin/env bash
# =============================================================================
# 25_storage_bench.sh — 量三種儲存的頻寬/延遲，決定論文「SSD 階」要掛在哪
#
# 為什麼非量不可（見 docs/MI300X_MLSTEAM.md §7.5.5）：
#   平台 A（3090）的 SSD 階是 /ssd7，**本地 SSD**。
#   平台 B 這裡持久化的掛載點（workspace / data）**都是 NFS over TCP**，
#   只有 `/` 的 overlay 是本地 SSD（但它不持久，而且是整台 host 共用）。
#   直接拿 NFS 當「SSD 階」會把網路成本混進 κ，跨硬體的歸因就毀了。
#
#   → 所以先量，再決定，**不要先假設**。
#
# 量什麼：KV cache spill 的存取形態是「大塊、循序、一次寫完再整批讀」，
#   所以主測 seq write / seq read @ 1 MiB；順便量 4 KiB 隨機延遲當參考。
#
#   用法： bash 25_storage_bench.sh [每項測試的檔案大小，預設 4G]
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
DATA=/mlsteam/data/tiara
SIZE="${1:-4G}"
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
RUN="${TIARA_RUNS:-$DATA/runs}/$STAMP-storage-bench"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/stdout.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

command -v fio >/dev/null || { echo "[info] 裝 fio"; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fio; }

echo "=== 25_storage_bench.sh @ $(date -Is) ==="
echo "檔案大小 per test: $SIZE"
fio --version

# 三個候選路徑（第三個是 tmpfs，當作「記憶體速度」的上界參考）
declare -A TARGETS=(
  [nfs_data]="$DATA/.fio_bench"
  [nfs_workspace]="$WS/.fio_bench"
  [local_overlay]="/var/tmp/.fio_bench"
  [tmpfs_shm]="/dev/shm/.fio_bench"
)

CSV="$RUN/storage_bench.csv"
echo "run_id,ts,target,path,fstype,direct_ok,pattern,bs,bw_MBps,iops,lat_us_mean,lat_us_p99" > "$CSV"

for name in nfs_data nfs_workspace local_overlay tmpfs_shm; do
    DIR="${TARGETS[$name]}"
    mkdir -p "$DIR"
    FSTYPE=$(df -T "$DIR" | awk 'NR==2{print $2}')

    # 🔴 O_DIRECT 能力探測。
    #    overlayfs / tmpfs 不見得支援 O_DIRECT；不支援時 fio 的 --direct=1
    #    會被**靜默忽略**，讀取全部從 page cache 出來，數字會虛高好幾倍。
    #    第一次量到 overlay 循序讀 10.9 GB/s 而寫只有 2.7 GB/s，就是這個形狀。
    #    不驗證就報數字 = 報錯的數字。
    DIRECT_OK=$(python3 - "$DIR" <<'PYD'
import os, sys
p = os.path.join(sys.argv[1], ".odirect_probe")
try:
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
    os.close(fd); os.unlink(p); print("yes")
except OSError as e:
    try: os.unlink(p)
    except OSError: pass
    print(f"no({e.errno}:{e.strerror})")
except AttributeError:
    print("no(no-O_DIRECT-on-this-python)")
PYD
)
    echo
    echo "───────────────────────────────────────────────"
    echo "  $name  →  $DIR   (fstype=$FSTYPE, O_DIRECT=$DIRECT_OK)"
    if [ "$DIRECT_OK" != "yes" ]; then
        echo "  ⚠ 不支援 O_DIRECT → 讀取數字會被 page cache 灌水，**不可採信**"
    fi
    echo "───────────────────────────────────────────────"

    for spec in "seqwrite:write:1m" "seqread:read:1m" "randwrite:randwrite:4k" "randread:randread:4k"; do
        IFS=: read -r label rw bs <<< "$spec"
        J="$RUN/fio-$name-$label.json"

        # 🔴 讀取測試前，先把這個檔從 page cache 踢掉。
        #    為什麼不能只靠 --direct=1：O_DIRECT 的 open() 成功**不代表**真的
        #    繞過快取。證據是 tmpfs（本身就是記憶體）也回報 O_DIRECT=yes，
        #    而第一版量到 overlay 循序讀 12.0 GB/s > tmpfs 的 10.4 GB/s ——
        #    磁碟贏記憶體是物理上不可能的，那個數字來自 page cache。
        #    posix_fadvise(POSIX_FADV_DONTNEED) 才會真的丟掉該檔的 cache page。
        if [[ "$label" == *read* ]] && [ -f "$DIR/testfile" ]; then
            python3 - "$DIR/testfile" <<'PYF'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
PYF
            sync
        fi
        # --direct=1 繞過 page cache；NFS 上 direct IO 才量得到真正的網路成本。
        # --end_fsync=1 確保寫入真的落地，不是留在 client 端快取裡。
        fio --name="$label" --filename="$DIR/testfile" --size="$SIZE" \
            --rw="$rw" --bs="$bs" --direct=1 --end_fsync=1 \
            --ioengine=libaio --iodepth=16 --numjobs=1 \
            --runtime=30 --time_based=0 --group_reporting \
            --output-format=json --output="$J" >/dev/null 2>&1 || {
                echo "  [warn] $label 失敗（有些 fs 不支援 direct IO），記錄後繼續"
                echo "$STAMP,$(date -Is),$name,$DIR,$FSTYPE,$DIRECT_OK,$label,$bs,FAILED,FAILED,FAILED,FAILED" >> "$CSV"
                continue
            }
        python3 - "$J" "$CSV" "$STAMP" "$name" "$DIR" "$FSTYPE" "$DIRECT_OK" "$label" "$bs" <<'PYP'
import json, sys, datetime
j, csv, run_id, name, path, fstype, direct_ok, label, bs = sys.argv[1:10]
d = json.load(open(j))["jobs"][0]
sec = d["write"] if "write" in label else d["read"]
bw   = sec["bw_bytes"] / 1e6
iops = sec["iops"]

# fio 把 percentile 放在 clat_ns（completion latency），lat_ns 那份常常沒有
# percentile 子欄位 —— 上一版只看 lat_ns，於是 p99 全部讀成 0.0。
# 依序試 clat_ns -> lat_ns -> clat（舊版 fio 用微秒）。
mean = p99 = 0.0
for key, scale in (("clat_ns", 1000.0), ("lat_ns", 1000.0), ("clat", 1.0)):
    blk = sec.get(key)
    if not isinstance(blk, dict):
        continue
    pct = blk.get("percentile", {})
    if pct:
        for pk in ("99.000000", "99.00", "99"):
            if pk in pct:
                p99 = pct[pk] / scale
                break
    if blk.get("mean"):
        mean = blk["mean"] / scale
    if p99:
        break
if not p99:
    p99 = float("nan")     # 寧可寫 nan 也不要寫 0.0 假裝量到了

with open(csv, "a") as f:
    f.write(f"{run_id},{datetime.datetime.now().astimezone().isoformat()},{name},{path},"
            f"{fstype},{direct_ok},{label},{bs},{bw:.1f},{iops:.0f},{mean:.1f},{p99:.1f}\n")
flag = "" if direct_ok == "yes" else "  ⚠cached"
print(f"  {label:<10} bw={bw:8.1f} MB/s  iops={iops:9.0f}  lat_mean={mean:8.1f}us  p99={p99:9.1f}us{flag}")
PYP
    done
    rm -f "$DIR/testfile"
    rmdir "$DIR" 2>/dev/null || true
done

echo
echo "=== 結果 ==="
column -s, -t < "$CSV"
echo
echo "CSV: $CSV"
cat <<'EOF'

判讀指引：
  * 若 local_overlay 的 seqwrite 明顯高於 nfs_*（預期會），
    論文的「SSD 階」就用 local_overlay 上的 scratch 目錄，並在報告寫明：
      - 它是 overlayfs on LVM on SATA/NVMe SSD，**不是** raw NVMe
      - sda 是整台 host 共用，量測期間要用 gpulog/iostat 記錄爭用
  * **tmpfs_shm 那一列是對照組**，不是候選載體。它代表「純記憶體速度」的上界。
    任何 fstype 的讀取若 >= tmpfs，必然是量到快取而非裝置 —— 直接判定該數字無效。
  * **先看 direct_ok 欄**。不是 yes 的那幾列，讀取頻寬一律不可採信
    （--direct=1 被靜默忽略，量到的是 page cache）。寫入因為有 end_fsync
    比較可信，但也只是「比較」。
  * 若 NFS 與本地接近，代表 NFS 也被快取蓋掉了，重量。
  * 任何一欄 FAILED 都不要忽略，先弄清楚為什麼才下結論。
EOF
