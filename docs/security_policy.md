# Security Policy

## Secrets
- Never commit secrets to git.
- Use environment variables or a secret manager.

## Access
- Apply least privilege for all services.
- Rotate keys every 90 days or immediately after suspected compromise.

## Logging
- Do not log PII in plaintext.
- Redact tokens, passwords, and session identifiers.
