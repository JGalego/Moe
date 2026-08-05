# Assets

Every recording here is produced from a tape in `tapes/`, against a real
checkpoint — a demo of a random-weight fixture would be a demo of nothing.
`sh scripts/record.sh` regenerates them all; `sh scripts/record.sh route eval`
does a subset. It needs [vhs](https://github.com/charmbracelet/vhs), the release
binary on `PATH`, and about 12 GB of weights, mostly OLMoE-1B-7B.

| asset | tape | shows |
| --- | --- | --- |
| `run.gif` | `run.tape` | generating from a Hub repo id |
| `info.gif` | `info.tape` | architecture detected before and after packing |
| `gguf.gif` | `gguf.tape` | reading a GGUF checkpoint in place |
| `serve.gif` | `serve.tape` | chat, embeddings and metrics from one process |
| `chat.gif` | `chat.tape` | a terminal conversation, reusing the cache |
| `draft.gif` | `draft.tape` | speculative decoding: same output, fewer steps |
| `json.gif` | `json.tape` | decoding constrained to a JSON Schema |
| `route.gif` | `route.tape` | the routing analysis, and the diff below |
| `routing.svg` | `route.tape` | code versus prose, per expert per layer |
| `prune.gif` | `prune.tape` | tracing a domain, then pruning to it |
| `eval.gif` | `eval.tape` | what Q4 costs, as perplexity |
| `embed.gif` | `embed.tape` | the same checkpoint as an embedding model |

`logo.svg` is drawn by hand, not recorded.
