# Generation straight from a Hugging Face repo id.
#   moe pull allenai/OLMoE-1B-7B-0924   (once, before recording)
#   vhs tapes/run.tape
Source common.tape
Output assets/run.gif
Set Width 1240
Set Height 210

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe run allenai/OLMoE-1B-7B-0924 -p 'The largest planet in our solar system is' -n 26 --stats"
Sleep 700ms
Enter
Sleep 16s
