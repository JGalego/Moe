#!/usr/bin/env sh
# Record every demo in tapes/ into assets/.
#
# Needs vhs (https://github.com/charmbracelet/vhs), the release binary on PATH,
# and the models the tapes use — they are recorded against real checkpoints on
# purpose, because a demo of a tiny random-weight fixture would be a demo of
# nothing. Roughly 12 GB of weights and half an hour, mostly downloading.
#
#     cargo build --release && export PATH="$PWD/target/release:$PATH"
#     sh scripts/record.sh              # everything
#     sh scripts/record.sh route eval   # just these
set -eu

BASE=allenai/OLMoE-1B-7B-0924
INSTRUCT=allenai/OLMoE-1B-7B-0924-Instruct

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "$1 is not on PATH"; exit 1; }
}
need vhs
need moe

# Fixtures the tapes read. Built once; delete them to rebuild.
prepare() {
    echo "== pulling weights (cached after the first run)"
    moe pull "$BASE"
    moe pull "$INSTRUCT"

    [ -f olmoe-q4.moe ] || {
        echo "== packing a Q4 copy, for the eval comparison"
        moe pack "$BASE" --quant q8 --expert-quant q4 -o olmoe-q4.moe
    }
    [ -f wiki.txt ] || {
        echo "== assembling held-out text to score"
        # Any plain prose will do; the point is that the model has not been
        # given the answers. Substitute your own corpus freely.
        moe --help > wiki.txt
        for _ in 1 2 3 4 5; do cat README.md >> wiki.txt; done
    }
    [ -f person.json ] || cat > person.json <<'JSON'
{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"admin":{"type":"boolean"}},"required":["name","age","admin"]}
JSON
}

prepare

TAPES=${*:-}
if [ -z "$TAPES" ]; then
    TAPES=$(ls tapes/*.tape | sed 's|tapes/||; s|\.tape$||' | grep -v '^common$')
fi

for t in $TAPES; do
    echo "== recording $t"
    vhs "tapes/$t.tape"
done
echo "== done; assets/ updated"
