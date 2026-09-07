
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("USER_DB_PATH", "data/qx_users.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TOKEN_TTL = 12 * 60 * 60
_SESSIONS = {}
_ADMIN_SESSIONS = {}


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                max_devices INTEGER NOT NULL DEFAULT 1,
                device_id TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        db.commit()


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return salt, digest.hex()


def _check_password(password, salt, expected):
    _, actual = _hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _parse_dt(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _row_user(row):
    if not row:
        return None
    expires = _parse_dt(row["expires_at"])
    return {
        "id": row["id"],
        "username": row["username"],
        "expires_at": row["expires_at"],
        "max_devices": row["max_devices"],
        "device_bound": bool(row["device_id"]),
        "active": bool(row["active"]),
        "expired": expires <= _now(),
    }


def create_user(username, password, days, max_devices=1):
    username = username.strip().lower()
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    if days < 1 or days > 3650:
        raise ValueError("Days must be between 1 and 3650")
    if max_devices < 1 or max_devices > 5:
        raise ValueError("max_devices must be between 1 and 5")

    salt, digest = _hash_password(password)
    now = _now()
    expires = now + timedelta(days=days)
    try:
        with _db() as db:
            cur = db.execute(
                "INSERT INTO users(username,password_hash,password_salt,expires_at,max_devices,created_at) VALUES(?,?,?,?,?,?)",
                (username, digest, salt, _iso(expires), max_devices, _iso(now)),
            )
            db.commit()
            return get_user(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise ValueError("Username already exists")


def get_user(user_id):
    with _db() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _row_user(row)


def list_users():
    with _db() as db:
        rows = db.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    return [_row_user(r) for r in rows]


def login_user(username, password, device_id=None):
    username = username.strip().lower()
    with _db() as db:
        row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

    if not row or not _check_password(password, row["password_salt"], row["password_hash"]):
        raise ValueError("Invalid username or password")
    if not row["active"]:
        raise PermissionError("Account is suspended")
    if _parse_dt(row["expires_at"]) <= _now():
        raise PermissionError("Subscription expired")

    device_id = (device_id or "").strip()
    if device_id:
        bound = row["device_id"]
        if bound and bound != device_id and row["max_devices"] <= 1:
            raise PermissionError("Account is already bound to another device")
        if not bound:
            with _db() as db:
                db.execute("UPDATE users SET device_id=? WHERE id=?", (device_id, row["id"]))
                db.commit()

    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = {"user_id": row["id"], "created": time.time()}
    return token, get_user(row["id"])


def require_user(token):
    if not token:
        raise PermissionError("Authentication required")
    session = _SESSIONS.get(token)
    if not session:
        raise PermissionError("Invalid session")
    if time.time() - session["created"] > TOKEN_TTL:
        _SESSIONS.pop(token, None)
        raise PermissionError("Session expired")
    user = get_user(session["user_id"])
    if not user or not user["active"]:
        raise PermissionError("Account is suspended")
    if user["expired"]:
        raise PermissionError("Subscription expired")
    return user


def admin_login(password):
    expected = os.getenv("ADMIN_PASSWORD", "")
    if not expected:
        raise RuntimeError("ADMIN_PASSWORD environment variable is not configured")
    if not hmac.compare_digest(str(password), expected):
        raise ValueError("Invalid admin password")
    token = secrets.token_urlsafe(32)
    _ADMIN_SESSIONS[token] = {"created": time.time()}
    return token


def require_admin(token):
    if not token:
        raise PermissionError("Admin authentication required")
    session = _ADMIN_SESSIONS.get(token)
    if not session:
        raise PermissionError("Invalid admin session")
    if time.time() - session["created"] > TOKEN_TTL:
        _ADMIN_SESSIONS.pop(token, None)
        raise PermissionError("Admin session expired")


def extend_user(user_id, days):
    if days < 1 or days > 3650:
        raise ValueError("Days must be between 1 and 3650")
    with _db() as db:
        row = db.execute("SELECT expires_at FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            raise ValueError("User not found")
        old = _parse_dt(row["expires_at"])
        base = max(old, _now())
        expires = base + timedelta(days=days)
        db.execute("UPDATE users SET expires_at=?, active=1 WHERE id=?", (_iso(expires), user_id))
        db.commit()
    return get_user(user_id)


def set_active(user_id, active):
    with _db() as db:
        cur = db.execute("UPDATE users SET active=? WHERE id=?", (1 if active else 0, user_id))
        if cur.rowcount == 0:
            raise ValueError("User not found")
        db.commit()
    return get_user(user_id)


init_db()
