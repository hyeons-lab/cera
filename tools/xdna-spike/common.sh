# Shared settings for baselines.sh and npu_power.sh (sourced, not run). Run the
# scripts from WSL; every measured program runs natively on Windows. Each
# default can be overridden through the environment.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
win() { wslpath -w "$1"; }

PY="${PY:-py.exe}"                                        # Windows Python 3 (the py launcher)
SM="${SM:-$(win "$HERE/hwinfo_sm.py")}"
SPIKE="${SPIKE:-$(win "$HERE/spike.exe")}"                # built by build.ps1
ARTIFACTS="${ARTIFACTS:-$(win "$HERE/out")}"              # iron/build.sh output (OUT_DIR)
CERA="${CERA:-$(win "$REPO/target/release/cera.exe")}"    # cargo build --release -p cera-cli --features gpu
if [ -z "${MD:-}" ]; then
  MD="$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')\\.leap\\models"
fi
OUT="${OUT:-$HERE/results}"
model() {  # model <quant>: Windows path of the Llama-3.2-1B Instruct GGUF at that quant
  echo "$MD\\Llama-3.2-1B-Instruct-$1\\Llama-3.2-1B-Instruct-$1.gguf"
}
mkdir -p "$OUT"
OUT_WIN="$(win "$OUT")"

sampled() {  # sampled <tag> <hwinfo_sm.py run options...> -- <command...>: one measurement, logged
  local tag=$1; shift
  "$PY" "$SM" run --csv "$OUT_WIN\\$tag.csv" "$@" 2>&1 | tr -d '\r' | tee "$OUT/$tag.log"
}

need_file() {  # need_file <windows path> <what>
  [ -e "$(wslpath -u "$1")" ] || { echo "missing $2: $1 (override through the environment; see README.md)" >&2; exit 1; }
}
need_file "$SM" "hwinfo_sm.py"
