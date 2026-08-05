# What Q4 actually costs, as a number rather than a compression ratio.
#   moe pull allenai/OLMoE-1B-7B-0924
#   moe pack allenai/OLMoE-1B-7B-0924 --quant q8 --expert-quant q4 -o olmoe-q4.moe
#   head -c 200000 <some plain text> > wiki.txt
#   vhs tapes/eval.tape
Source common.tape
Output assets/eval.gif
Set Width 1240
Set Height 300

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe eval allenai/OLMoE-1B-7B-0924 --text wiki.txt --ctx 512 --vs olmoe-q4.moe"
Sleep 700ms
Enter
Sleep 40s
