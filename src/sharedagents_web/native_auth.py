"""Native sharedagents web authentication — the PRIMARY auth path.

sharedagents issues and verifies its OWN credentials against a local SQLite store,
with NO dependency on any external system.  This is the source of truth for who
may access the web UI; dashboard ``/verify-token`` delegation is at most an
optional alternative called external auth (see ``server.require_auth``, which is native-first).

Design (spec/web-ui.md §4):

  * Operator accounts live in ``users(email, password_hash, created_at,
    disabled)``.  Email stays the identity key because the whole ownership
    model already scopes on ``g.user_email``.
  * Login sessions are opaque 256-bit CSPRNG tokens, stored ONLY as SHA-256
    hashes in ``auth_tokens`` and delivered as an ``HttpOnly; Secure;
    SameSite=Lax`` cookie.  A DB read therefore never yields a usable bearer
    credential, and a token cannot be forged without the server's RNG output.
  * Verification touches nothing but local SQLite — so valid native credentials
    authenticate even with every external dependency unreachable, and
    a switch to native-primary never weakens the existing gate (fail-closed:
    absent/invalid/expired/revoked -> deny).

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library.  The spec's
v1 target is argon2id (argon2-cffi); it is a drop-in upgrade on this same schema
and is deferred here only to avoid adding a runtime dependency — the stored
format is self-describing (``algo$...``) so a later upgrade can re-hash on next
login without a migration.

Each account also carries a ``role`` (``admin`` | ``user``) and the ``grants``
(agents / models / tools / skills, :mod:`sharedagents_core.access_grants`) an admin
gave it.  Admins are the single operator; users are scoped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from sharedagents_core.access_grants import ALL, DIMENSIONS, Grants
from sharedagents_session.actors import clean_display_name, display_name_fallback

# Cookie carrying the native login token.  Distinct from any dashboard cookie so
# the two paths never collide.
COOKIE_NAME = "sharedagents_session"

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLES = (ROLE_ADMIN, ROLE_USER)


def env_admin_emails() -> set[str]:
    """``SHAREDAGENTS_WEB_ADMINS`` as a lowercase set (comma-separated)."""
    raw = os.environ.get("SHAREDAGENTS_WEB_ADMINS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def local_operator_email() -> str:
    """``SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL``, lowercased ('' when unset)."""
    return os.environ.get("SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL", "").strip().lower()


def resolve_admin_source(store: NativeAuthStore, email: str) -> str | None:
    """Why ``email`` is an admin — ``env`` (``SHAREDAGENTS_WEB_ADMINS``),
    ``operator`` (``SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL``), ``role`` (the account's
    role), ``fallback`` (no enabled admin exists, and this is the oldest enabled
    account, so the installation is never locked out) — or ``None``.

    Store-level so every transport (web, realtime voice) decides alike; a
    transport's own debug bypass is its caller's business.
    """
    e = (email or "").strip().lower()
    if not e:
        return None
    acct = store.account(e)
    if acct is not None and acct["disabled"]:
        return None
    if e in env_admin_emails():
        return "env"
    operator = local_operator_email()
    if operator and e == operator:
        return "operator"
    enabled = acct is not None and not acct["disabled"]
    if enabled and acct["role"] == ROLE_ADMIN:
        return "role"
    if enabled and not _any_enabled_admin(store) and e == (store.first_enabled_user() or "").lower():
        return "fallback"
    return None


def _any_enabled_admin(store: NativeAuthStore) -> bool:
    if store.enabled_role_admins():
        return True
    for email in env_admin_emails() | {local_operator_email()}:
        acct = store.account(email) if email else None
        if acct is not None and not acct["disabled"]:
            return True
    return False


class LastAdminError(ValueError):
    """The change would leave no enabled admin."""


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}")
    return role


# Login token lifetime — long-lived by default (single-operator tool on operator
# devices; a PWA makes re-login friction expensive).  Sliding on last_seen.
TOKEN_TTL_DAYS = 90

_PBKDF2_ITERATIONS = 240_000
_PBKDF2_ALGO = "pbkdf2_sha256"


def _now() -> int:
    return int(time.time())


def hash_password(password: str, *, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Return a self-describing ``algo$iters$salt$hash`` string."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_PBKDF2_ALGO}${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify of ``password`` against a stored hash string."""
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$", 3)
    except (ValueError, AttributeError):
        return False
    if algo != _PBKDF2_ALGO:
        return False
    try:
        iterations = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    email         TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0,
    display_name  TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT 'user',
    grants        TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash   TEXT PRIMARY KEY,
    email        TEXT NOT NULL,
    device_label TEXT,
    created_at   INTEGER NOT NULL,
    last_seen    INTEGER NOT NULL,
    expires_at   INTEGER NOT NULL,
    revoked      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_email ON auth_tokens(email);
"""


def _ensure_user_columns(db: sqlite3.Connection) -> None:
    """Additive, idempotent migration for pre-attribution ``users`` tables.

    The web DB has no ``user_version`` (three independent stores share the
    file), so — like ``loop_registry._ensure_loop_option_columns`` — probe
    ``PRAGMA table_info`` and ALTER only when the column is missing. SQLite
    ADD COLUMN with a constant default rewrites no rows (O(1)).
    """
    cols = {str(row[1]) for row in db.execute("PRAGMA table_info(users)")}
    if "display_name" not in cols:
        try:
            db.execute("ALTER TABLE users ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError as exc:  # another gunicorn worker won the race
            if "duplicate column" not in str(exc).lower():
                raise
    if "role" not in cols or "grants" not in cols:
        _add_access_columns(db)
    _backfill_folder_grants(db)


def _backfill_folder_grants(db: sqlite3.Connection) -> None:
    """Give All folders to All grants stored before folder grants existed.

    A missing ``folders`` key reads as no folders (fail closed), so an
    account the role upgrade gave All would otherwise wake up confined to
    nothing. Only a stored All (every dimension ``"*"``) gains
    ``"folders": "*"``; any other grant keeps no folders. Every write stores
    the key, so this is idempotent; the compare-and-set never overwrites a
    grant an admin changed meanwhile.
    """
    for email, raw in db.execute("SELECT email, grants FROM users").fetchall():
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict) and "folders" not in obj and all(obj.get(dim) == ALL for dim in DIMENSIONS):
            db.execute(
                "UPDATE users SET grants=? WHERE email=? AND grants=?",
                (Grants.all().to_json(), email, raw),
            )


def _add_access_columns(db: sqlite3.Connection) -> None:
    """Add ``role`` + ``grants`` and classify the accounts that predate them.

    Before roles every account was the operator. When
    ``SHAREDAGENTS_WEB_ADMINS`` names the admins, those accounts and the local
    operator stay admins and everyone else becomes a ``user`` with All
    grants — nothing they could use is taken away, but their chats become
    private. Without the env var nobody can tell operators apart, so every
    existing account stays an admin. Runs once: the classification is
    tied to adding the ``role`` column, inside one IMMEDIATE transaction.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        cols = {str(row[1]) for row in db.execute("PRAGMA table_info(users)")}
        if "grants" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN grants TEXT NOT NULL DEFAULT '{}'")
        if "role" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            admins = env_admin_emails()
            if admins:
                admins.add(local_operator_email())
                emails = [str(r[0]) for r in db.execute("SELECT email FROM users")]
                for email in emails:
                    if email.lower() in admins:
                        db.execute("UPDATE users SET role=? WHERE email=?", (ROLE_ADMIN, email))
                    else:
                        db.execute(
                            "UPDATE users SET grants=? WHERE email=?",
                            (Grants.all().to_json(), email),
                        )
            else:
                db.execute("UPDATE users SET role=?", (ROLE_ADMIN,))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise


class NativeAuthStore:
    """SQLite-backed native credential store (users + hashed login tokens)."""

    def __init__(self, db_path):
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._reopen_gate = threading.Lock()
        self._inherited_conns: list[sqlite3.Connection] = []
        self._pid = os.getpid()
        self._open_db()
        self._db.executescript(_SCHEMA_SQL)
        _ensure_user_columns(self._db)

    def _open_db(self) -> None:
        self._db = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")

    def reopen_after_fork(self) -> None:
        """Abandon a SQLite connection inherited across Gunicorn's fork.

        Do not ``close()`` the inherited handle (SQLAlchemy
        ``dispose(close=False)``). Pin it so GC cannot close it either.
        """
        with self._reopen_gate:
            if self._pid == os.getpid():
                return
            self._inherited_conns.append(self._db)
            self._open_db()
            self._db.executescript(_SCHEMA_SQL)
            _ensure_user_columns(self._db)
            self._pid = os.getpid()

    def _acquire(self) -> threading.RLock:
        if os.getpid() != self._pid:
            self.reopen_after_fork()
        return self._lock

    # -- accounts -----------------------------------------------------------

    def create_user(
        self,
        email: str,
        password: str,
        *,
        replace: bool = False,
        display_name: str | None = None,
        role: str | None = None,
        grants: Grants | None = None,
    ) -> None:
        """Create an account (``replace`` = CLI password reset / recovery).

        A new account is a ``user`` with no grants unless ``role`` / ``grants``
        say otherwise. A reset keeps the stored role and grants unless they
        are passed explicitly.
        """
        email = (email or "").strip().lower()
        if not email:
            raise ValueError("email required")
        if role is not None:
            _check_role(role)
        pw_hash = hash_password(password)
        name = clean_display_name(display_name) if display_name is not None else ""
        new_role = role or ROLE_USER
        new_grants = (grants or Grants.none()).to_json()
        with self._acquire():
            if replace:
                # UPSERT, not INSERT OR REPLACE: REPLACE deletes the row and would
                # wipe display_name on every CLI password reset. password_hash /
                # created_at / disabled keep exactly their previous semantics.
                self._db.execute(
                    "INSERT INTO users(email, password_hash, created_at, disabled, display_name, role, grants) "
                    "VALUES(?,?,?,0,?,?,?) ON CONFLICT(email) DO UPDATE SET "
                    "password_hash=excluded.password_hash, created_at=excluded.created_at, disabled=0",
                    (email, pw_hash, _now(), name, new_role, new_grants),
                )
                if display_name is not None:
                    self._db.execute(
                        "UPDATE users SET display_name=? WHERE email=?",
                        (name, email),
                    )
                if role is not None:
                    self._db.execute("UPDATE users SET role=? WHERE email=?", (role, email))
                if grants is not None:
                    self._db.execute("UPDATE users SET grants=? WHERE email=?", (new_grants, email))
            else:
                self._db.execute(
                    "INSERT INTO users(email, password_hash, created_at, disabled, display_name, role, grants) "
                    "VALUES(?,?,?,0,?,?,?)",
                    (email, pw_hash, _now(), name, new_role, new_grants),
                )
            self._db.commit()

    # -- roles and grants -----------------

    @staticmethod
    def _account_row(row) -> dict:
        return {
            "email": str(row["email"]),
            "display_name": clean_display_name(row["display_name"]),
            "role": str(row["role"] or ROLE_USER),
            "grants": Grants.from_json(row["grants"]),
            "disabled": bool(row["disabled"]),
            "created_at": int(row["created_at"] or 0),
            "last_seen": int(row["last_seen"] or 0) if "last_seen" in row.keys() else 0,  # noqa: SIM118 - sqlite3.Row "in" checks values
        }

    def account(self, email: str) -> dict | None:
        """Role, grants and status of one account, or ``None``."""
        email = (email or "").strip().lower()
        if not email:
            return None
        with self._acquire():
            row = self._db.execute(
                "SELECT email, display_name, role, grants, disabled, created_at FROM users WHERE email=?",
                (email,),
            ).fetchone()
        return self._account_row(row) if row is not None else None

    def list_accounts(self) -> list[dict]:
        """Every account, oldest first, with its latest sign-in activity."""
        with self._acquire():
            rows = self._db.execute(
                "SELECT u.email, u.display_name, u.role, u.grants, u.disabled, "
                "u.created_at, (SELECT MAX(t.last_seen) FROM auth_tokens t "
                "WHERE t.email = u.email) AS last_seen "
                "FROM users u ORDER BY u.created_at ASC, u.email ASC"
            ).fetchall()
        return [self._account_row(r) for r in rows]

    def set_role(self, email: str, role: str) -> bool:
        _check_role(role)
        email = (email or "").strip().lower()
        with self._acquire():
            cur = self._db.execute("UPDATE users SET role=? WHERE email=?", (role, email))
            self._db.commit()
            return cur.rowcount > 0

    def set_grants(self, email: str, grants: Grants) -> bool:
        email = (email or "").strip().lower()
        with self._acquire():
            cur = self._db.execute(
                "UPDATE users SET grants=? WHERE email=?",
                (grants.to_json(), email),
            )
            self._db.commit()
            return cur.rowcount > 0

    def set_disabled(self, email: str, disabled: bool) -> bool:
        """Disable (every login revoked at once) or re-enable an account."""
        email = (email or "").strip().lower()
        with self._acquire():
            cur = self._db.execute(
                "UPDATE users SET disabled=? WHERE email=?",
                (1 if disabled else 0, email),
            )
            if disabled:
                self._db.execute("UPDATE auth_tokens SET revoked=1 WHERE email=?", (email,))
            self._db.commit()
            return cur.rowcount > 0

    def update_account(
        self,
        email: str,
        *,
        role: str | None = None,
        disabled: bool | None = None,
        grants: Grants | None = None,
        display_name: str | None = None,
    ) -> bool:
        """Apply an admin's edits in ONE write transaction.

        The last-admin guard runs inside the same ``BEGIN IMMEDIATE``, so two
        concurrent demotions (two gunicorn workers) cannot both see "another
        admin remains" and leave nobody. Raises :class:`LastAdminError`
        (nothing written) when no enabled admin would remain. Disabling
        revokes every login. False when the account does not exist.
        """
        email = (email or "").strip().lower()
        if role is not None:
            _check_role(role)
        with self._acquire():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT 1 FROM users WHERE email=?",
                    (email,),
                ).fetchone()
                if row is None:
                    self._db.execute("ROLLBACK")
                    return False
                if display_name is not None:
                    self._db.execute(
                        "UPDATE users SET display_name=? WHERE email=?",
                        (clean_display_name(display_name), email),
                    )
                if role is not None:
                    self._db.execute("UPDATE users SET role=? WHERE email=?", (role, email))
                if grants is not None:
                    self._db.execute(
                        "UPDATE users SET grants=? WHERE email=?",
                        (grants.to_json(), email),
                    )
                if disabled is not None:
                    self._db.execute(
                        "UPDATE users SET disabled=? WHERE email=?",
                        (1 if disabled else 0, email),
                    )
                    if disabled:
                        self._db.execute(
                            "UPDATE auth_tokens SET revoked=1 WHERE email=?",
                            (email,),
                        )
                if role is not None or disabled is not None:
                    env = env_admin_emails() | {local_operator_email()}
                    left = sum(
                        1
                        for r in self._db.execute("SELECT email, role, disabled FROM users")
                        if not r["disabled"] and (r["role"] == ROLE_ADMIN or r["email"] in env)
                    )
                    if left == 0:
                        self._db.execute("ROLLBACK")
                        raise LastAdminError("At least one admin must remain.")
                self._db.execute("COMMIT")
            except LastAdminError:
                raise
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return True

    def token_active(self, raw_token: str) -> str | None:
        """The owner of a still-valid login token, without sliding its expiry
        (for re-checks on long-lived streams); ``None`` when revoked, expired
        or disabled."""
        if not raw_token or not isinstance(raw_token, str):
            return None
        with self._acquire():
            row = self._db.execute(
                "SELECT t.email AS email, t.revoked AS revoked, t.expires_at AS expires_at, "
                "u.disabled AS disabled FROM auth_tokens t JOIN users u ON u.email = t.email "
                "WHERE t.token_hash=?",
                (_token_hash(raw_token),),
            ).fetchone()
        if row is None or row["revoked"] or row["disabled"] or row["expires_at"] <= _now():
            return None
        return str(row["email"])

    def revoke_all_tokens(self, email: str) -> int:
        email = (email or "").strip().lower()
        with self._acquire():
            cur = self._db.execute(
                "UPDATE auth_tokens SET revoked=1 WHERE email=? AND revoked=0",
                (email,),
            )
            self._db.commit()
            return cur.rowcount

    def enabled_role_admins(self) -> list[str]:
        """Enabled accounts whose stored role is admin, oldest first."""
        with self._acquire():
            rows = self._db.execute(
                "SELECT email FROM users WHERE role=? AND disabled=0 ORDER BY created_at ASC, email ASC",
                (ROLE_ADMIN,),
            ).fetchall()
        return [str(r["email"]) for r in rows]

    def display_name(self, email: str) -> str:
        """Display name for ``email``: stored value, else the email local part."""
        email = (email or "").strip().lower()
        if not email:
            return ""
        with self._acquire():
            row = self._db.execute(
                "SELECT display_name FROM users WHERE email=?",
                (email,),
            ).fetchone()
        stored = clean_display_name(row["display_name"]) if row is not None else ""
        return stored or display_name_fallback(email)

    def set_display_name(self, email: str, name: str) -> bool:
        """Set (``""`` clears) a user's display name. False when no such user."""
        email = (email or "").strip().lower()
        with self._acquire():
            cur = self._db.execute(
                "UPDATE users SET display_name=? WHERE email=?",
                (clean_display_name(name), email),
            )
            self._db.commit()
            return cur.rowcount > 0

    def set_password(self, email: str, password: str) -> bool:
        email = (email or "").strip().lower()
        pw_hash = hash_password(password)
        with self._acquire():
            cur = self._db.execute("UPDATE users SET password_hash=? WHERE email=?", (pw_hash, email))
            self._db.commit()
            return cur.rowcount > 0

    def user_exists(self, email: str) -> bool:
        email = (email or "").strip().lower()
        with self._acquire():
            row = self._db.execute("SELECT 1 FROM users WHERE email=? AND disabled=0", (email,)).fetchone()
        return row is not None

    def has_any_user(self) -> bool:
        with self._acquire():
            row = self._db.execute("SELECT 1 FROM users WHERE disabled=0 LIMIT 1").fetchone()
        return row is not None

    def first_enabled_user(self) -> str | None:
        """Return the first native operator account, deterministically.

        The Web→gateway principal mapper uses this only when no explicit
        ``SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL`` is configured.  Its result is
        persisted in the Web principal-binding table, so later account
        creation cannot change an established installation owner.
        """
        with self._acquire():
            row = self._db.execute(
                "SELECT email FROM users WHERE disabled=0 ORDER BY created_at ASC, email ASC LIMIT 1"
            ).fetchone()
        return str(row["email"]) if row is not None else None

    def verify_credentials(self, email: str, password: str) -> bool:
        email = (email or "").strip().lower()
        with self._acquire():
            row = self._db.execute(
                "SELECT password_hash FROM users WHERE email=? AND disabled=0",
                (email,),
            ).fetchone()
        if row is None:
            # Spend a hash anyway to blunt user-enumeration timing.
            verify_password(password or "", hash_password("x"))
            return False
        return verify_password(password or "", row["password_hash"])

    # -- login tokens -------------------------------------------------------

    def issue_token(
        self,
        email: str,
        *,
        device_label: str | None = None,
        ttl_days: int = TOKEN_TTL_DAYS,
    ) -> str:
        """Mint a new login token for ``email`` and return the RAW token.

        Only the SHA-256 hash is persisted; the raw value is returned once, to
        be set as the client cookie.
        """
        email = (email or "").strip().lower()
        raw = secrets.token_urlsafe(32)  # 256 bits of entropy
        now = _now()
        expires = now + ttl_days * 86400
        with self._acquire():
            self._db.execute(
                "INSERT INTO auth_tokens(token_hash, email, device_label, "
                "created_at, last_seen, expires_at, revoked) VALUES(?,?,?,?,?,?,0)",
                (_token_hash(raw), email, device_label, now, now, expires),
            )
            self._db.commit()
        return raw

    def verify_token(self, raw_token: str) -> str | None:
        """Return the owning email for a valid token, else ``None``.

        Fail-closed on: unknown, revoked, expired, or belonging to a disabled/
        missing user.  Refreshes ``last_seen`` (sliding expiry) on success.  No
        network, no external dependency — pure local SQLite.
        """
        if not raw_token or not isinstance(raw_token, str):
            return None
        th = _token_hash(raw_token)
        now = _now()
        with self._acquire():
            row = self._db.execute(
                "SELECT t.email AS email, t.revoked AS revoked, "
                "t.expires_at AS expires_at, u.disabled AS disabled "
                "FROM auth_tokens t JOIN users u ON u.email = t.email "
                "WHERE t.token_hash=?",
                (th,),
            ).fetchone()
            if row is None or row["revoked"] or row["disabled"]:
                return None
            if row["expires_at"] <= now:
                return None
            # Sliding refresh: bump last_seen and extend expiry from now.
            self._db.execute(
                "UPDATE auth_tokens SET last_seen=?, expires_at=? WHERE token_hash=?",
                (now, now + TOKEN_TTL_DAYS * 86400, th),
            )
            self._db.commit()
        return row["email"]

    def revoke_token(self, raw_token: str) -> bool:
        if not raw_token:
            return False
        with self._acquire():
            cur = self._db.execute(
                "UPDATE auth_tokens SET revoked=1 WHERE token_hash=?",
                (_token_hash(raw_token),),
            )
            self._db.commit()
            return cur.rowcount > 0

    def list_tokens(self, email: str):
        email = (email or "").strip().lower()
        with self._acquire():
            rows = self._db.execute(
                "SELECT token_hash, device_label, created_at, last_seen, "
                "expires_at, revoked FROM auth_tokens WHERE email=? "
                "ORDER BY last_seen DESC",
                (email,),
            ).fetchall()
        return [dict(r) for r in rows]
