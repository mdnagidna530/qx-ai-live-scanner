import hashlib
import hmac
import os
import secrets
import sqlite3
import time

from datetime import datetime, timedelta, timezone
from pathlib import Path


# =========================================================
# DATABASE CONFIG
# =========================================================

DB_PATH = Path(
    os.getenv(
        "USER_DB_PATH",
        "data/qx_users.db",
    )
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# Customer/Admin session lifetime
# 12 hours
TOKEN_TTL = 12 * 60 * 60


# =========================================================
# DATABASE
# =========================================================

def _db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    return conn


# =========================================================
# TIME HELPERS
# =========================================================

def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(
        timezone.utc
    ).isoformat()


def _parse_dt(value):
    return datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )


# =========================================================
# TOKEN HELPERS
# =========================================================

def _token_hash(token):
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def _new_token():
    return secrets.token_urlsafe(32)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():

    with _db() as db:

        # -------------------------------------------------
        # CUSTOMER ACCOUNTS
        # -------------------------------------------------
        #
        # max_devices and device_id are intentionally kept
        # in the SQLite schema for compatibility with an
        # existing database created by an older version.
        #
        # They are NO LONGER USED for authentication.
        #
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

        # -------------------------------------------------
        # CUSTOMER SESSIONS
        # -------------------------------------------------

        db.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (

                token_hash TEXT PRIMARY KEY,

                user_id INTEGER NOT NULL,

                created_at REAL NOT NULL,

                expires_at REAL NOT NULL

            )
        """)

        # -------------------------------------------------
        # ADMIN SESSIONS
        # -------------------------------------------------

        db.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (

                token_hash TEXT PRIMARY KEY,

                created_at REAL NOT NULL,

                expires_at REAL NOT NULL

            )
        """)

        db.commit()


# =========================================================
# SESSION CLEANUP
# =========================================================

def _cleanup_sessions():

    now = time.time()

    with _db() as db:

        db.execute(
            """
            DELETE FROM user_sessions
            WHERE expires_at <= ?
            """,
            (now,),
        )

        db.execute(
            """
            DELETE FROM admin_sessions
            WHERE expires_at <= ?
            """,
            (now,),
        )

        db.commit()


# =========================================================
# PASSWORD HASHING
# =========================================================

def _hash_password(
    password,
    salt=None,
):

    salt = salt or secrets.token_hex(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200_000,
    )

    return (
        salt,
        digest.hex(),
    )


def _check_password(
    password,
    salt,
    expected,
):

    _, actual = _hash_password(
        password,
        salt,
    )

    return hmac.compare_digest(
        actual,
        expected,
    )


# =========================================================
# USER ROW -> API OBJECT
# =========================================================

def _row_user(row):

    if not row:
        return None

    expires = _parse_dt(
        row["expires_at"]
    )

    return {
        "id": row["id"],

        "username": row["username"],

        "expires_at": row["expires_at"],

        # Kept only for API/database compatibility.
        # Device restriction is NOT enforced.
        "max_devices": row["max_devices"],

        # Always false because device binding
        # has been disabled.
        "device_bound": False,

        "active": bool(
            row["active"]
        ),

        "expired": (
            expires <= _now()
        ),
    }


# =========================================================
# USER MANAGEMENT
# =========================================================

def create_user(
    username,
    password,
    days,
    max_devices=1,
):

    username = (
        username
        .strip()
        .lower()
    )

    if not username or len(username) < 3:

        raise ValueError(
            "Username must be at least 3 characters"
        )

    if len(password) < 6:

        raise ValueError(
            "Password must be at least 6 characters"
        )

    if days < 1 or days > 3650:

        raise ValueError(
            "Days must be between 1 and 3650"
        )

    # -----------------------------------------------------
    # Device restriction is disabled.
    #
    # max_devices is accepted only so the existing
    # admin API does not break.
    # -----------------------------------------------------

    salt, digest = _hash_password(
        password
    )

    now = _now()

    expires = (
        now
        + timedelta(
            days=days
        )
    )

    try:

        with _db() as db:

            cur = db.execute(
                """
                INSERT INTO users(
                    username,
                    password_hash,
                    password_salt,
                    expires_at,
                    max_devices,
                    device_id,
                    active,
                    created_at
                )
                VALUES(
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    NULL,
                    1,
                    ?
                )
                """,
                (
                    username,
                    digest,
                    salt,
                    _iso(expires),

                    # Keep legacy DB field harmless.
                    # It has NO authentication effect.
                    1,

                    _iso(now),
                ),
            )

            db.commit()

            return get_user(
                cur.lastrowid
            )

    except sqlite3.IntegrityError:

        raise ValueError(
            "Username already exists"
        )


def get_user(user_id):

    with _db() as db:

        row = db.execute(
            """
            SELECT *
            FROM users
            WHERE id=?
            """,
            (user_id,),
        ).fetchone()

    return _row_user(
        row
    )


def list_users():

    with _db() as db:

        rows = db.execute(
            """
            SELECT *
            FROM users
            ORDER BY id DESC
            """
        ).fetchall()

    return [
        _row_user(row)
        for row in rows
    ]


# =========================================================
# CUSTOMER LOGIN
# =========================================================

def login_user(
    username,
    password,
    device_id=None,
):

    _cleanup_sessions()

    username = (
        username
        .strip()
        .lower()
    )

    with _db() as db:

        row = db.execute(
            """
            SELECT *
            FROM users
            WHERE username=?
            """,
            (username,),
        ).fetchone()

    # -----------------------------------------------------
    # Username/password check
    # -----------------------------------------------------

    if not row:

        raise ValueError(
            "Invalid username or password"
        )

    if not _check_password(
        password,
        row["password_salt"],
        row["password_hash"],
    ):

        raise ValueError(
            "Invalid username or password"
        )

    # -----------------------------------------------------
    # Active check
    # -----------------------------------------------------

    if not row["active"]:

        raise PermissionError(
            "Account is suspended"
        )

    # -----------------------------------------------------
    # Subscription expiry check
    # -----------------------------------------------------

    if (
        _parse_dt(
            row["expires_at"]
        )
        <= _now()
    ):

        raise PermissionError(
            "Subscription expired"
        )

    # -----------------------------------------------------
    # DEVICE BINDING DISABLED
    # -----------------------------------------------------
    #
    # device_id is intentionally ignored.
    #
    # The same username/password can be used from
    # different devices.
    #
    # No Android ID is required.
    # No device binding is performed.
    # No device limit is enforced.
    # -----------------------------------------------------

    token = _new_token()

    token_hash = _token_hash(
        token
    )

    now = time.time()

    expires = (
        now
        + TOKEN_TTL
    )

    with _db() as db:

        db.execute(
            """
            INSERT INTO user_sessions(
                token_hash,
                user_id,
                created_at,
                expires_at
            )
            VALUES(
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                token_hash,
                row["id"],
                now,
                expires,
            ),
        )

        db.commit()

    return (
        token,
        get_user(
            row["id"]
        ),
    )


# =========================================================
# CUSTOMER SESSION VALIDATION
# =========================================================

def require_user(token):

    _cleanup_sessions()

    if not token:

        raise PermissionError(
            "Authentication required"
        )

    token_hash = _token_hash(
        token
    )

    with _db() as db:

        session = db.execute(
            """
            SELECT *
            FROM user_sessions
            WHERE token_hash=?
            """,
            (token_hash,),
        ).fetchone()

    if not session:

        raise PermissionError(
            "Invalid session"
        )

    if (
        time.time()
        >= session["expires_at"]
    ):

        # Remove expired session
        with _db() as db:

            db.execute(
                """
                DELETE FROM user_sessions
                WHERE token_hash=?
                """,
                (token_hash,),
            )

            db.commit()

        raise PermissionError(
            "Session expired"
        )

    user = get_user(
        session["user_id"]
    )

    if not user:

        raise PermissionError(
            "Invalid session"
        )

    # -----------------------------------------------------
    # Account active check
    # -----------------------------------------------------

    if not user["active"]:

        raise PermissionError(
            "Account is suspended"
        )

    # -----------------------------------------------------
    # Subscription check
    # -----------------------------------------------------

    if user["expired"]:

        raise PermissionError(
            "Subscription expired"
        )

    return user


# =========================================================
# ADMIN LOGIN
# =========================================================

def admin_login(password):

    _cleanup_sessions()

    expected = os.getenv(
        "ADMIN_PASSWORD",
        "",
    ).strip()

    if not expected:

        raise RuntimeError(
            "ADMIN_PASSWORD environment variable is not configured"
        )

    if not hmac.compare_digest(
        str(password),
        expected,
    ):

        raise ValueError(
            "Invalid admin password"
        )

    token = _new_token()

    token_hash = _token_hash(
        token
    )

    now = time.time()

    expires = (
        now
        + TOKEN_TTL
    )

    with _db() as db:

        db.execute(
            """
            INSERT INTO admin_sessions(
                token_hash,
                created_at,
                expires_at
            )
            VALUES(
                ?,
                ?,
                ?
            )
            """,
            (
                token_hash,
                now,
                expires,
            ),
        )

        db.commit()

    return token


# =========================================================
# ADMIN SESSION VALIDATION
# =========================================================

def require_admin(token):

    _cleanup_sessions()

    if not token:

        raise PermissionError(
            "Admin authentication required"
        )

    token_hash = _token_hash(
        token
    )

    with _db() as db:

        session = db.execute(
            """
            SELECT *
            FROM admin_sessions
            WHERE token_hash=?
            """,
            (token_hash,),
        ).fetchone()

    if not session:

        raise PermissionError(
            "Invalid admin session"
        )

    if (
        time.time()
        >= session["expires_at"]
    ):

        with _db() as db:

            db.execute(
                """
                DELETE FROM admin_sessions
                WHERE token_hash=?
                """,
                (token_hash,),
            )

            db.commit()

        raise PermissionError(
            "Admin session expired"
        )


# =========================================================
# SUBSCRIPTION MANAGEMENT
# =========================================================

def extend_user(
    user_id,
    days,
):

    if days < 1 or days > 3650:

        raise ValueError(
            "Days must be between 1 and 3650"
        )

    with _db() as db:

        row = db.execute(
            """
            SELECT expires_at
            FROM users
            WHERE id=?
            """,
            (user_id,),
        ).fetchone()

        if not row:

            raise ValueError(
                "User not found"
            )

        old = _parse_dt(
            row["expires_at"]
        )

        base = max(
            old,
            _now(),
        )

        expires = (
            base
            + timedelta(
                days=days
            )
        )

        db.execute(
            """
            UPDATE users
            SET expires_at=?,
                active=1
            WHERE id=?
            """,
            (
                _iso(expires),
                user_id,
            ),
        )

        db.commit()

    return get_user(
        user_id
    )


# =========================================================
# ACTIVATE / SUSPEND USER
# =========================================================

def set_active(
    user_id,
    active,
):

    with _db() as db:

        cur = db.execute(
            """
            UPDATE users
            SET active=?
            WHERE id=?
            """,
            (
                1 if active else 0,
                user_id,
            ),
        )

        if cur.rowcount == 0:

            raise ValueError(
                "User not found"
            )

        db.commit()

    return get_user(
        user_id
    )


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_db()
