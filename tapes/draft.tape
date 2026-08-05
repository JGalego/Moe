# Speculative decoding: identical output, fewer forward passes.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/draft.tape
Source common.tape
Output assets/draft.gif
Set Width 1240
Set Height 400

Hide
Type "PS1='$ ' && clear" Enter
Type "P='Repeat this list back exactly: alpha, beta, gamma, delta, epsilon. The list is: alpha, beta, gamma, delta, epsilon. Again: '" Enter
Type "clear" Enter
Show

Type "moe run allenai/OLMoE-1B-7B-0924-Instruct -p \"$P\" -n 32 --no-stream"
Sleep 700ms
Enter
Sleep 12s

Type "moe run allenai/OLMoE-1B-7B-0924-Instruct -p \"$P\" -n 32 --no-stream --draft 8"
Sleep 700ms
Enter
Sleep 14s
