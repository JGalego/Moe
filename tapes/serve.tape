# The server: chat, embeddings and metrics from one process.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/serve.tape
Source common.tape
Output assets/serve.gif
Set Width 1240
Set Height 480

Hide
Type "PS1='$ ' && clear" Enter
Show

Type "moe serve allenai/OLMoE-1B-7B-0924-Instruct --slots 4 --draft 8 &"
Sleep 500ms
Enter
Sleep 8s

Type "curl -s localhost:8080/v1/chat/completions -d '{\"messages\":[{\"role\":\"user\",\"content\":\"One word for a small stream?\"}],\"max_tokens\":12}' | jq -r .choices[0].message.content"
Sleep 500ms
Enter
Sleep 12s

Type "curl -s localhost:8080/v1/embeddings -d '{\"input\":\"a small stream\"}' | jq '.data[0].embedding[:4]'"
Sleep 500ms
Enter
Sleep 6s

Type "curl -s localhost:8080/metrics | grep -E 'prefix_cache|draft_' | grep -v '^#'"
Sleep 500ms
Enter
Sleep 4s
