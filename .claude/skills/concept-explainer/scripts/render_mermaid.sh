#!/usr/bin/env bash
# 문서 안의 Mermaid 블록을 모두 PNG로 렌더링해서 문법 오류와 모양을 확인한다.
# 사용법: render_mermaid.sh 문서.md [출력폴더]
# 출력폴더 기본값: ./mermaid-out
# mermaid-cli가 없으면 임시 폴더에 설치한다. 브라우저 경로는 PUPPETEER_EXECUTABLE_PATH로 지정할 수 있다.

set -eu
doc="${1:?사용법: $0 문서.md [출력폴더]}"
out="${2:-./mermaid-out}"
mkdir -p "$out"

# 1. Mermaid 블록 추출
python3 - "$doc" "$out" <<'PY'
import re, sys, pathlib
doc, out = sys.argv[1], pathlib.Path(sys.argv[2])
blocks = re.findall(r'```mermaid\n(.*?)```', open(doc, encoding='utf-8').read(), re.S)
for i, b in enumerate(blocks):
    (out / f"diagram-{i:02d}.mmd").write_text(b, encoding='utf-8')
print(f"Mermaid 블록 {len(blocks)}개")
PY

# 2. mmdc 준비
if command -v mmdc >/dev/null 2>&1; then
  MMDC=mmdc
else
  tool_dir="${TMPDIR:-/tmp}/concept-explainer-mmdc"
  if [ ! -x "$tool_dir/node_modules/.bin/mmdc" ]; then
    echo "mermaid-cli 설치 중: $tool_dir"
    mkdir -p "$tool_dir"
    (cd "$tool_dir" && npm init -y >/dev/null && PUPPETEER_SKIP_DOWNLOAD="${PUPPETEER_EXECUTABLE_PATH:+1}" npm install --silent @mermaid-js/mermaid-cli >/dev/null)
  fi
  MMDC="$tool_dir/node_modules/.bin/mmdc"
fi

# 3. 브라우저 설정 (지정된 경로가 없으면 흔한 위치를 찾아본다)
if [ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
  for c in /opt/pw-browsers/chromium-*/chrome-linux/chrome; do
    [ -x "$c" ] && PUPPETEER_EXECUTABLE_PATH="$c" && break
  done
fi
cfg="$out/puppeteer.json"
if [ -n "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
  printf '{"executablePath":"%s","args":["--no-sandbox"]}\n' "$PUPPETEER_EXECUTABLE_PATH" > "$cfg"
else
  printf '{"args":["--no-sandbox"]}\n' > "$cfg"
fi

# 4. 렌더링
fail=0
for f in "$out"/diagram-*.mmd; do
  [ -e "$f" ] || continue
  if "$MMDC" -p "$cfg" -i "$f" -o "${f%.mmd}.png" -b white -s 1.5 >/dev/null 2>"${f%.mmd}.err"; then
    echo "OK   ${f%.mmd}.png"
    rm -f "${f%.mmd}.err"
  else
    echo "FAIL $f"; sed 's/^/     /' "${f%.mmd}.err" | head -5
    fail=$((fail + 1))
  fi
done
echo "실패 ${fail}개. PNG를 열어 상자 폭, 줄바꿈, 화살표 엉킴을 눈으로 확인한다."
[ "$fail" -eq 0 ]
