# The server: chat, embeddings and metrics from one process.
#   moe pull allenai/OLMoE-1B-7B-0924-Instruct
#   vhs tapes/serve.tape
Source tapes/common.tape
Output assets/serve.gif
Set Width 1240
Set Height 440

Hide
Type "PS1='$ ' && clear" Enter
Type "cat > ask.json <<'JSON'" Enter
Type '{"messages":[{"role":"user","content":"One word for a small stream?"}],' Enter
Type '"max_tokens":12}' Enter
Type "JSON" Enter
Type "cat > list.json <<'JSON'" Enter
Type '{"messages":[{"role":"user","content":"Repeat this list back exactly, once: alpha, beta, gamma, delta, epsilon."}],' Enter
Type '"max_tokens":24}' Enter
Type "JSON" Enter
Type "clear" Enter
Show

Type "moe serve allenai/OLMoE-1B-7B-0924-Instruct --slots 4 --draft 8 &"
Sleep 500ms
Enter
Sleep 12s
Enter
Sleep 500ms

Type "curl -s localhost:8080/v1/chat/completions -d @ask.json | jq -r .choices[0].message.content"
Sleep 500ms
Enter
Sleep 14s

Type "curl -s localhost:8080/v1/chat/completions -d @list.json | jq -r .choices[0].message.content"
Sleep 500ms
Enter
Sleep 16s

Type `curl -s localhost:8080/v1/embeddings -d '{"input": "a small stream"}' | jq -c .data[0].embedding[:4]`
Sleep 500ms
Enter
Sleep 10s

Type "curl -s localhost:8080/metrics | grep -E 'draft_|prefix_cache_tokens' | grep -v '^#'"
Sleep 500ms
Enter
Sleep 5s
