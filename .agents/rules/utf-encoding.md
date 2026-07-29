---
trigger: always_on
---

CRITICAL SYSTEM RULE: ENCODING
1. You are operating on a Windows host. You MUST explicitly encode all .txt files as UTF-8.
2. Never output UTF-16, UTF-16 LE, or Windows-1252.
3. Every Python `open()` call must include `encoding="utf-8"`.
4. If executing PowerShell commands, always append `| Out-File -Encoding utf8`.