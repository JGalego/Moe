#!/usr/bin/env sh
# Record every demo in tapes/ into assets/.
#
# Needs vhs (https://github.com/charmbracelet/vhs) with ttyd and ffmpeg, the
# release binary on PATH, and the checkpoint the tapes use — they are recorded
# against a real model on purpose, because a demo of a tiny random-weight fixture
# would be a demo of nothing. About 20 GB of weights: OLMoE-1B-7B-Instruct for
# every tape, plus its Q4_0 GGUF for gguf.tape, both fetched on first use.
#
#     cargo build --release && export PATH="$PWD/target/release:$PATH"
#     sh scripts/record.sh              # everything
#     sh scripts/record.sh route eval   # just these
set -eu

# One checkpoint drives every tape: it is a capable base LM for run, route, eval,
# embed and prune, and it ships a chat template, which chat and serve need.
MODEL=allenai/OLMoE-1B-7B-0924-Instruct

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "$1 is not on PATH"; exit 1; }
}
need vhs
need moe
need curl # serve.tape talks to the server it starts
need jq

echo "== pulling weights (cached after the first run)"
moe pull "$MODEL" >/dev/null

# The one fixture a tape cannot build for itself: the Q4 copy `eval` compares
# against. The rest — the schema, the request bodies, the routing trace — are
# written by the tapes, in their own hidden setup blocks.
[ -f olmoe-q4.moe ] || {
    echo "== packing a Q4 copy, for the eval comparison"
    moe pack "$MODEL" --quant q8 --expert-quant q4 -o olmoe-q4.moe
}

TAPES=${*:-}
if [ -z "$TAPES" ]; then
    TAPES=$(ls tapes/*.tape | sed 's|tapes/||; s|\.tape$||' | grep -v '^common$')
fi

for t in $TAPES; do
    echo "== recording $t"
    vhs "tapes/$t.tape"
done
echo "== done; assets/ updated"
