# Where a model sends code versus prose — the routing analysis, in the binary.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct   (once, before recording)
#   vhs tapes/route.tape
Source tapes/common.tape
Output assets/route.gif
Set Width 1240
Set Height 500

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe route allenai/OLMoE-1B-7B-0924-Instruct --top 3 \"
Enter
Type "  -p 'def parse(xs): return [int(x) for x in xs.split()]' | head -8"
Sleep 700ms
Enter
Sleep 16s

Type "moe route allenai/OLMoE-1B-7B-0924-Instruct --top 3 \"
Enter
Type "  -p 'def parse(xs): return [int(x) for x in xs.split()]' \"
Enter
Type "  --vs 'The French Revolution began in 1789, when the Estates-General met' \"
Enter
Type "  -o assets/routing.svg"
Sleep 700ms
Enter
Sleep 18s
