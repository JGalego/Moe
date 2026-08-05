# A conversation in the terminal, reusing the cache between turns.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/chat.tape
Source common.tape
Output assets/chat.gif
Set Width 1240
Set Height 460
Set TypingSpeed 60ms

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe chat allenai/OLMoE-1B-7B-0924-Instruct --stats -n 60"
Sleep 700ms
Enter
Sleep 6s

Type "Name three cold-water fish."
Sleep 400ms
Enter
Sleep 20s

Type "And one of those in fresh water?"
Sleep 400ms
Enter
Sleep 20s

Type "/exit"
Enter
Sleep 1s
