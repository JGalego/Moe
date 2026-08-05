# Architecture detected from the checkpoint, before and after packing.
#   moe pack allenai/OLMoE-1B-7B-0924-Instruct --quant q8 --expert-quant q4 -o olmoe-q4.moe
#   vhs tapes/info.tape
Source tapes/common.tape
Output assets/info.gif
Set Width 1240
Set Height 520

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe info allenai/OLMoE-1B-7B-0924-Instruct"
Sleep 700ms
Enter
Sleep 2500ms

Type "moe info olmoe-q4.moe"
Sleep 700ms
Enter
Sleep 3500ms
