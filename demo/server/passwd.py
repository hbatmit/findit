"""Password hashing for the demo's shared password.

    python -m server.passwd          # prompts twice, prints the config.toml line

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


def _main() -> None:
    import getpass

    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Again: "):
        sys.exit("passwords differ")
    if len(pw) < 8:
        sys.exit("use at least 8 characters")
    print("\nPut this line under [server] in config.toml:\n")
    print(f'password_hash = "{hash_password(pw)}"')


if __name__ == "__main__":
    _main()
