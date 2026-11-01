"""Kalshi request signing (docs.kalshi.com/getting_started/api_keys, read 2026-10-29).

Pre-sign text = timestamp_ms + HTTP method + path from the API root, WITHOUT the query
string (e.g. "1730000000000POST/trade-api/v2/portfolio/events/orders"). Ed25519 keys sign
it directly; RSA keys use RSA-PSS (SHA-256, MGF1-SHA-256, salt = digest length).
"""

import base64
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa


class CredentialError(RuntimeError):
    pass


def load_private_key(path: str):
    p = Path(path).expanduser()
    if not path or not p.is_file():
        raise CredentialError(f"private key file not found: {path!r}")
    repo = Path(__file__).resolve().parents[1]
    if repo in p.resolve().parents:
        raise CredentialError("refusing to load a private key stored inside the repository; "
                              "move it outside the project folder")
    key = serialization.load_pem_private_key(p.read_bytes(), password=None)
    if not isinstance(key, (ed25519.Ed25519PrivateKey, rsa.RSAPrivateKey)):
        raise CredentialError(f"unsupported key type {type(key).__name__}")
    return key


def presign_text(timestamp_ms: int, method: str, url: str) -> str:
    return f"{timestamp_ms}{method.upper()}{urlparse(url).path}"


def sign(key, text: str) -> str:
    msg = text.encode("utf-8")
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(msg)
    else:
        sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                        salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return base64.b64encode(sig).decode("ascii")


def auth_headers(key_id: str, key, method: str, url: str, timestamp_ms: int) -> dict:
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "KALSHI-ACCESS-SIGNATURE": sign(key, presign_text(timestamp_ms, method, url)),
    }
