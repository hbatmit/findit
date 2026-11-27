"""Password hashing for the demo's shared password.

    python -m server.passwd          # prompts twice, prints the config.toml line
    python -m server.passwd --write  # same, and writes it into config.toml (FINDIT_CONFIG)

Format: pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>. Standard library only.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import sys

ALGO = "pbkdf2_sha256"
ITERATIONS = 600_000


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return "$".join([ALGO, str(iterations), base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_b64, hash_b64 = stored.split("$")
        if algo != ALGO:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iterations))
        return secrets.compare_digest(digest, base64.b64decode(hash_b64))
    except (ValueError, TypeError):
        return False


def write_hash(path, line: str) -> None:
    """Set password_hash under [server] in a TOML file, replacing any existing one."""
    import re
    from pathlib import Path

    path = Path(path)
    text = path.read_text() if path.exists() else ""
    if re.search(r"^password_hash\s*=", text, re.M):
        text = re.sub(r"^password_hash\s*=.*$", line, text, count=1, flags=re.M)
    elif re.search(r"^\[server\]", text, re.M):
        text = re.sub(r"^\[server\].*$", lambda m: m.group(0) + "\n" + line, text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n\n[server]\n" + line + "\n"
    path.write_text(text)


def _main() -> None:
    import getpass
    import os

    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Again: "):
        sys.exit("passwords differ")
    if len(pw) < 8:
        sys.exit("use at least 8 characters")
    line = f'password_hash = "{hash_password(pw)}"'
    if "--write" in sys.argv:
        path = os.environ.get("FINDIT_CONFIG", "config.toml")
        write_hash(path, line)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        print(f"wrote password_hash to {path} (mode 600). Restart the server to apply it.")
    else:
        print("\nPut this line under [server] in config.toml:\n")
        print(line)


if __name__ == "__main__":
    _main()
