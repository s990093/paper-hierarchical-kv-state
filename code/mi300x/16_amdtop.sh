#!/usr/bin/env bash
# =============================================================================
# 16_amdtop.sh — 裝 amdtop（nvitop 風格的 AMD GPU TUI）
#
#   https://github.com/lhl/amdtop
#   「nvitop-style TUI frontend for amdgpu_top (AMDGPU + Strix Halo XDNA NPU)」
#
# 為什麼要它：rocm-smi 是表格、nvtop 偏簡略。amdtop 是 nvitop/btop 風格，
#   一畫面同時看 CPU 每核心、GPU 頻寬/功耗/溫度、以及**per-process 的 VRAM**。
#   最後那項對本專案特別重要：量測時要能一眼確認「這張卡上只有我」。
#
# 版本（2026-09-15 實查）：
#   amdtop crates.io max_stable = 0.2.6（2026-07-23）
#   需求 Rust >= 1.88；目前 stable 是 1.98.1 → 沒問題
#   需求 libdrm 開發標頭（Debian/Ubuntu: libdrm-dev）
#
# Rust 工具鏈與 cargo 產物**全部放 workspace**，跟 node/claude 一樣跨 Lab 存活，
# 不必每次重開都重編（cargo 在 NFS 上編很慢，編一次就好）。
# =============================================================================
set -Eeuo pipefail
WS=/mlsteam/workspace
export RUSTUP_HOME="$WS/rust/rustup"
export CARGO_HOME="$WS/rust/cargo"
BIN_ROOT="$WS/rust/bin-root"          # cargo install --root
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
exec > >(tee -a "$LOG_DIR/amdtop-$STAMP.log") 2>&1
trap 'echo "[FATAL] line $LINENO 失敗，exit=$?" >&2' ERR

AMDTOP_VERSION="${AMDTOP_VERSION:-0.2.6}"

echo "=== 16_amdtop.sh @ $(date -Is) ==="

# --- 0. 前置條件（缺了就停，不要編到一半才爆）--------------------------------
missing=0
if [ ! -e /dev/kfd ]; then echo "[FATAL] 沒有 /dev/kfd —— 這個容器看不到 AMD GPU" >&2; missing=1; fi
if [ ! -d /dev/dri ]; then echo "[FATAL] 沒有 /dev/dri" >&2; missing=1; fi
[ "$missing" = 0 ] || exit 1
echo "[ok] /dev/kfd 與 /dev/dri 都在：$(ls /dev/dri | tr '\n' ' ')"

export DEBIAN_FRONTEND=noninteractive
dpkg -s libdrm-dev >/dev/null 2>&1 || {
    echo "[info] 裝 libdrm-dev（amdtop 編譯需要）"
    apt-get update -qq && apt-get install -y -qq libdrm-dev pkg-config
}
ldconfig -p | grep -q libdrm_amdgpu || {
    echo "[FATAL] 找不到 libdrm_amdgpu.so.1（amdtop 的唯一 runtime 相依）" >&2; exit 1; }
echo "[ok] libdrm 就緒"

# --- 1. Rust 工具鏈（裝進 workspace）-----------------------------------------
mkdir -p "$RUSTUP_HOME" "$CARGO_HOME" "$BIN_ROOT"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    echo "[info] 安裝 Rust 工具鏈 -> $CARGO_HOME"
    # rustup 官方安裝器。放 workspace 是為了跨 Lab 存活，不是為了省事。
    curl -fsSL --proto '=https' --tlsv1.2 https://sh.rustup.rs -o /tmp/rustup-init.sh
    sha256sum /tmp/rustup-init.sh          # 留在 log 備查
    sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain stable
    rm -f /tmp/rustup-init.sh
else
    echo "[skip] Rust 已在 $CARGO_HOME"
fi
export PATH="$CARGO_HOME/bin:$BIN_ROOT/bin:$PATH"
RUSTV=$(rustc --version)
echo "[ok] $RUSTV"
# 需求 >= 1.88，用資料自己驗證而不是假設
rustc --version | awk '{split($2,v,"."); if (v[1]<1 || (v[1]==1 && v[2]<88)) {
    print "[FATAL] rustc " $2 " < 1.88，amdtop 編不過"; exit 1 }}' || exit 1

# --- 2. amdtop ---------------------------------------------------------------
# 🔴 不能直接 `cargo install amdtop` 就了事 —— crates.io 的版本在這個容器裡
#    一啟動就 panic。原因不是我們裝錯，是 upstream 的假設在共用叢集上不成立：
#
#      host 有 8 張 MI300X，/sys/bus/pci/drivers/amdgpu/ 看得到全部 9 個 BDF；
#      但容器的 /dev/dri 只有配給我們的 card41 / renderD168（0000:a6:00.0）。
#      libamdgpu_top 的 DevicePath::get_fd() 開不到節點時是 panic!() 不是 Err，
#      所以 amdtop 死在第一張碰不到的卡（0000:06:00.0）上。
#
#    → 從 git 原始碼建，套一個「跳過節點不存在的裝置」的小 patch。
#    patch 內容見 code/mi300x/patches/amdtop-skip-inaccessible-devices.patch
SRC_DIR="$WS/src/amdtop"
PATCH_MARK="skipped .* GPU(s) visible in sysfs"

need_build=1
if [ -x "$BIN_ROOT/bin/amdtop" ] && strings "$BIN_ROOT/bin/amdtop" 2>/dev/null | grep -q "visible in sysfs"; then
    need_build=0
    echo "[skip] 已安裝含 patch 的 amdtop"
fi

if [ "$need_build" = "1" ]; then
    mkdir -p "$WS/src"
    if [ -d "$SRC_DIR/.git" ]; then
        git -C "$SRC_DIR" fetch --all --tags -q
        git -C "$SRC_DIR" checkout -f -q "v$AMDTOP_VERSION" 2>/dev/null \
            || git -C "$SRC_DIR" checkout -f -q main
    else
        git clone -q https://github.com/lhl/amdtop "$SRC_DIR"
        git -C "$SRC_DIR" checkout -q "v$AMDTOP_VERSION" 2>/dev/null || true
    fi
    echo "[ok] amdtop 原始碼 @ $(git -C "$SRC_DIR" rev-parse --short HEAD)"

    # 套 patch（冪等：已套過就跳過）
    if grep -q "visible in sysfs" "$SRC_DIR/src/app.rs"; then
        echo "[skip] patch 已套用"
    else
        python3 - "$SRC_DIR/src/app.rs" <<'PYPATCH'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = """    for device_path in &mut device_paths {
        device_path.fill_amdgpu_device_name();
    }
"""
if anchor not in s:
    sys.exit("[FATAL] amdtop 的 discover_devices() 結構跟預期不同，patch 的錨點找不到。\n"
             "        上游可能改過。不要硬套，先去看 src/app.rs。")
ins = """
    // ── 本地修改（MLSteam / k8s 多租戶容器）─────────────────────────────
    // host 有 8 張 MI300X，sysfs 看得到全部，但容器的 /dev/dri 只暴露配給
    // 我們的那一張。libamdgpu_top 的 get_fd() 開不到節點時是 panic!() 而非
    // Err，所以 amdtop 會死在第一張碰不到的卡上。upstream 假設「機器上的
    // GPU 都是自己的」，在共用叢集不成立。
    // ⚠️ 不要改成補上缺的 /dev 節點 —— 那會顯示別的租戶的數據（靜默錯誤）。
    let before = device_paths.len();
    device_paths.retain(|d| d.render.exists() || d.accel.exists());
    if device_paths.len() != before {
        eprintln!(
            "amdtop: skipped {} GPU(s) visible in sysfs but not exposed to this container",
            before - device_paths.len()
        );
    }
"""
p.write_text(s.replace(anchor, anchor + ins, 1))
print("[ok] patch 已套用到 src/app.rs")
PYPATCH
    fi

    echo "[info] 編譯 patched amdtop（NFS 上約 1–2 分鐘）"
    cargo install --path "$SRC_DIR" --root "$BIN_ROOT" --locked --force
fi

# --- 3. PATH（兩個 HOME 都要，見 §7.5.3）-------------------------------------
cat > /etc/profile.d/97-rust.sh <<EOF
export RUSTUP_HOME=$RUSTUP_HOME
export CARGO_HOME=$CARGO_HOME
export PATH=$CARGO_HOME/bin:$BIN_ROOT/bin:\$PATH
EOF
chmod 644 /etc/profile.d/97-rust.sh
for f in /root/.bashrc "$WS/.bashrc"; do
    touch "$f" 2>/dev/null || continue
    grep -q "CARGO_HOME=$CARGO_HOME" "$f" 2>/dev/null || cat >> "$f" <<EOF
export RUSTUP_HOME=$RUSTUP_HOME
export CARGO_HOME=$CARGO_HOME
export PATH=$CARGO_HOME/bin:$BIN_ROOT/bin:\$PATH
EOF
done

echo
echo "=== 完成 ==="
echo "  amdtop : $("$BIN_ROOT/bin/amdtop" --version 2>&1 | head -1)"
echo "  位置   : $BIN_ROOT/bin/amdtop   （NFS，跨 Lab 存活）"
echo "  rust   : $RUSTV  ($CARGO_HOME)"
echo
echo "  用法： ssh amd -t amdtop        # 需要 TTY，所以 -t"
echo "         t / T 換主題（41 個），預設 tokyo-night"
