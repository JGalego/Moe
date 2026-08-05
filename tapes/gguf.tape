# A GGUF checkpoint, read in place — no conversion step.
#   vhs tapes/gguf.tape
Source tapes/common.tape
Output assets/gguf.gif
Set Width 1240
Set Height 440

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "G=https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_0.gguf"
Sleep 400ms
Enter
Sleep 500ms

Type "moe info $G"
Sleep 700ms
Enter
Sleep 20s

Type "moe run $G -n 24 -p 'A sparse model is fast because' --stats"
Sleep 700ms
Enter
Sleep 25s
