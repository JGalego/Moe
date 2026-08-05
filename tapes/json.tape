# Constrained decoding: the model cannot emit anything but this shape.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/json.tape
Source common.tape
Output assets/json.gif
Set Width 1240
Set Height 380

Hide
Type "PS1='$ ' && clear" Enter
Type "cat > person.json <<'JSON'" Enter
Type '{"type":"object","properties":{"name":{"type":"string"},' Enter
Type '"age":{"type":"integer"},"admin":{"type":"boolean"}},' Enter
Type '"required":["name","age","admin"]}' Enter
Type "JSON" Enter
Type "clear" Enter
Show

Type "cat person.json"
Sleep 500ms
Enter
Sleep 1500ms

Type "moe run allenai/OLMoE-1B-7B-0924-Instruct --schema person.json -n 40 \"
Enter
Type "  -p 'Ada Lovelace, 36, an administrator. As JSON: '"
Sleep 700ms
Enter
Sleep 14s
