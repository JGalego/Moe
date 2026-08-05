# Trace a domain, then keep only the experts it reaches.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/prune.tape
Source common.tape
Output assets/prune.gif
Set Width 1240
Set Height 440

Hide
Type "PS1='$ ' && clear" Enter
Type "CODE='def merge(a, b): return {**a, **b}  # dict union, py3.5+'" Enter
Type "clear" Enter
Show

Type "moe run allenai/OLMoE-1B-7B-0924-Instruct -p \"$CODE\" -n 48 --trace code.jsonl --no-stream"
Sleep 700ms
Enter
Sleep 20s

Type "moe pack allenai/OLMoE-1B-7B-0924-Instruct --quant q8 --expert-quant q4 \"
Enter
Type "  --keep-experts code.jsonl --keep 16 -o olmoe-code.moe | tail -3"
Sleep 700ms
Enter
Sleep 45s

Type "moe info olmoe-code.moe | head -6"
Sleep 700ms
Enter
Sleep 4s
