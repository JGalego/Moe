# What Q4 actually costs, as a number rather than a compression ratio.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   moe pack allenai/OLMoE-1B-7B-0924-Instruct --quant q8 --expert-quant q4 -o olmoe-q4.moe
#   vhs tapes/eval.tape
# The corpus is the opening of Pride and Prejudice (Project Gutenberg, public
# domain), checked in so the number is reproducible.
Source tapes/common.tape
Output assets/eval.gif
Set Width 1240
Set Height 250

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe eval allenai/OLMoE-1B-7B-0924-Instruct --text tests/fixtures/austen.txt \"
Enter
Type "  --ctx 512 --limit 512 --vs olmoe-q4.moe"
Sleep 700ms
Enter
Sleep 80s
