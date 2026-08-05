# The same checkpoint as an embedding model.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/embed.tape
Source common.tape
Output assets/embed.gif
Set Width 1240
Set Height 300

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe embed allenai/OLMoE-1B-7B-0924-Instruct --pool last \"
Enter
Type "  -p 'a cat sat on the mat' --vs 'the kitten rested on the rug'"
Sleep 700ms
Enter
Sleep 8s

Type "moe embed allenai/OLMoE-1B-7B-0924-Instruct --pool last \"
Enter
Type "  -p 'a cat sat on the mat' --vs 'quarterly earnings exceeded guidance'"
Sleep 700ms
Enter
Sleep 10s
