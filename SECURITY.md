# Security policy

Use the latest published MSCS release and review its changelog before upgrading.
Security fixes are developed on the current release line; older versions do
not have a separate maintenance commitment.

## Reporting

If private vulnerability reporting is enabled, use the repository's
[private advisory form](https://github.com/ElEscribanoSilente/MSC-Serial/security/advisories/new).
Otherwise, open an issue requesting a private contact channel without including
an exploit, credentials or sensitive input. Include affected versions, platform,
impact and a minimal reproduction when a private channel is available.

## Trust and resource boundaries

- Register only trusted classes. Their reconstruction, attribute, Enum and
  hash/equality hooks can execute application code.
- Require `hmac_key` when your application needs authenticated messages. CRC
  protects against accidental corruption, and offers no sender authentication.
- Set `max_size`, `max_depth` and `max_hash_work` for your workload. These are
  parser limits, not a hard cap on process RAM or CPU. Collision/equality cost
  and registered hooks are outside the hash work budget.
- Compressed data is decompressed within limits before inner HMAC verification.
- Optional NumPy/PyTorch packages and your distribution channel have their own
  security lifecycle. Test your resolved dependencies and keep them updated.

Regression and stress checks cover specific behaviors and inputs. They do not
provide a guarantee that the package, its dependencies or a deployment is free
of vulnerabilities. See README.md for supported formats and compatibility limits.
