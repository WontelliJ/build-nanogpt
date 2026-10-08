#!/usr/bin/env bash
# 문서의 문체 규칙 위반 후보를 찾는다. 코드 블록(``` ... ```) 안은 검사하지 않는다.
# 사용법: check_style.sh 문서.md [문서2.md ...]
# 걸린 항목이 모두 위반은 아니다. 목록 나열처럼 의도한 경우는 그대로 둬도 된다.

set -u
if [ $# -eq 0 ]; then
  echo "사용법: $0 문서.md [문서2.md ...]" >&2
  exit 1
fi

check() {
  # $1: 설명, $2: 정규식, $3: 파일(코드 블록 제외본)
  local hits
  hits=$(grep -nE "$2" "$3" || true)
  if [ -n "$hits" ]; then
    echo "  [$1]"
    echo "$hits" | sed 's/^/    /'
    total=$((total + $(echo "$hits" | wc -l)))
  fi
}

for f in "$@"; do
  echo "== $f"
  total=0
  tmp=$(mktemp)
  # 코드 블록 안의 줄은 빈 줄로 바꿔 줄 번호를 유지한다
  awk '/^```/{inblock=!inblock; print ""; next} {print (inblock ? "" : $0)}' "$f" > "$tmp"

  check "연결어미 뒤 쉼표"   '[가-힣]+(고|며|면|서|지만|면서|라면),' "$tmp"
  check "굵은 글씨"          '\*\*' "$tmp"
  check "긴 줄표"            '—' "$tmp"
  check "큰따옴표 강조"      '"[^"]*[가-힣][^"]*"' "$tmp"
  check "Q&A 형식"           '^(- )?(Q|A)\. ' "$tmp"
  check "장 번호 참조"       '\((이유는 )?[0-9]+(-[0-9]+)?장' "$tmp"
  check "AI 관용구"          '셈이다|핵심은 |중요한 것은 |잠깐|단순한 [^ ]+ 넘어|주목할 만하다|시사하는 바' "$tmp"
  check "명사형 종결"        '(없음|있음|함|됨|임)\.?$' "$tmp"

  rm -f "$tmp"
  echo "  후보 ${total}건"
done
