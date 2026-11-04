"""OAuth 2.1 authorization server backed by the Telegram login page.

Flow (what an MCP client such as claude.ai / Claude Desktop / Claude Code does):
  1. GET /.well-known/oauth-authorization-server  -> metadata (SDK)
  2. POST /register                                -> dynamic client registration (SDK -> us)
  3. GET /authorize?...PKCE...                     -> we park the request as a "transaction"
                                                     and send the browser to /login?txn=...
  4. The person logs into Telegram on our page; on success we mint an auth code
     bound to the transaction and redirect back to the client's redirect_uri.
  5. POST /token (code + code_verifier)            -> access + refresh token (SDK -> us)

Every access token maps to exactly one connected Telegram account (users.id).
"""
import time
from typing import Optional

from pydantic import AnyUrl
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from . import config, security
from .db import Database, get_db

SCOPE = "telegram"
PERSONAL_CLIENT_ID = "personal"


class UserAccessToken(AccessToken):
    user_id: int = 0


class UserRefreshToken(RefreshToken):
    user_id: int = 0
    pair_id: str = ""


class UserAuthCode(AuthorizationCode):
    user_id: int = 0


class Transaction:
    """A pending /authorize request waiting for the Telegram login to finish."""
    __slots__ = ("id", "client_id", "client_name", "params", "created")

    def __init__(self, id: str, client: OAuthClientInformationFull, params: AuthorizationParams):
        self.id = id
        self.client_id = client.client_id or ""
        self.client_name = client.client_name or self.client_id
        self.params = params
        self.created = time.time()


class TelegramOAuthProvider(OAuthAuthorizationServerProvider[UserAuthCode, UserRefreshToken, UserAccessToken]):
    def __init__(self, db: Optional[Database] = None):
        self._db = db
        self.transactions: dict[str, Transaction] = {}

    @property
    def db(self) -> Database:
        return self._db or get_db()

    # ---- clients ----------------------------------------------------------
    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        data = self.db.get_client(client_id)
        if data is None:
            return None
        return OAuthClientInformationFull.model_validate(data)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            client_info.client_id = security.new_token(16)
        self.db.save_client(client_info.client_id, client_info.model_dump(mode="json", exclude_none=True))

    # ---- authorize ---------------------------------------------------------
    def _sweep(self):
        cutoff = time.time() - config.AUTH_CODE_TTL
        for k, t in list(self.transactions.items()):
            if t.created < cutoff:
                self.transactions.pop(k, None)

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        self._sweep()
        if params.scopes and any(s != SCOPE for s in params.scopes):
            raise AuthorizeError("invalid_scope", f"Only the '{SCOPE}' scope is supported.")
        txn = Transaction(security.new_token(24), client, params)
        self.transactions[txn.id] = txn
        return f"{config.PUBLIC_BASE_URL}/login?txn={txn.id}"

    def get_transaction(self, txn_id: Optional[str]) -> Optional[Transaction]:
        self._sweep()
        return self.transactions.get(txn_id or "")

    def complete_transaction(self, txn_id: str, user_id: int) -> str:
        """Mint an authorization code for a finished login and return the redirect URL."""
        txn = self.transactions.pop(txn_id, None)
        if txn is None:
            raise ValueError("Authorization request expired. Start again from your MCP client.")
        p = txn.params
        code = security.new_token(32)
        self.db.save_auth_code(
            code, txn.client_id, user_id,
            {
                "scopes": [SCOPE],
                "code_challenge": p.code_challenge,
                "redirect_uri": str(p.redirect_uri),
                "redirect_uri_provided_explicitly": p.redirect_uri_provided_explicitly,
                "resource": p.resource,
            },
            time.time() + config.AUTH_CODE_TTL,
        )
        return construct_redirect_uri(str(p.redirect_uri), code=code, state=p.state)

    def deny_transaction(self, txn_id: str) -> Optional[str]:
        txn = self.transactions.pop(txn_id, None)
        if txn is None:
            return None
        return construct_redirect_uri(str(txn.params.redirect_uri), error="access_denied", state=txn.params.state)

    # ---- codes -------------------------------------------------------------
    async def load_authorization_code(self, client: OAuthClientInformationFull,
                                      authorization_code: str) -> Optional[UserAuthCode]:
        d = self.db.peek_auth_code(authorization_code)
        if d is None or d["client_id"] != client.client_id:
            return None
        return UserAuthCode(
            code=d["code"], scopes=d["scopes"], expires_at=d["expires_at"], client_id=d["client_id"],
            code_challenge=d["code_challenge"], redirect_uri=AnyUrl(d["redirect_uri"]),
            redirect_uri_provided_explicitly=d["redirect_uri_provided_explicitly"],
            resource=d.get("resource"), user_id=d["user_id"],
        )

    async def exchange_authorization_code(self, client: OAuthClientInformationFull,
                                          authorization_code: UserAuthCode) -> OAuthToken:
        d = self.db.pop_auth_code(authorization_code.code)   # single use
        if d is None:
            raise TokenError("invalid_grant", "authorization code already used or expired")
        if d["expires_at"] < time.time():
            raise TokenError("invalid_grant", "authorization code expired")
        return self._issue(client.client_id or "", d["user_id"], d["scopes"], resource=d.get("resource"))

    # ---- tokens ------------------------------------------------------------
    def _issue(self, client_id: str, user_id: int, scopes: list[str],
               resource: Optional[str] = None, label: Optional[str] = None) -> OAuthToken:
        pair = security.new_token(12)
        access = security.new_token(32)
        refresh = security.new_token(32)
        now = int(time.time())
        self.db.save_token(access, kind="access", client_id=client_id, user_id=user_id, scopes=scopes,
                           pair_id=pair, expires_at=now + config.ACCESS_TOKEN_TTL, label=label)
        self.db.save_token(refresh, kind="refresh", client_id=client_id, user_id=user_id, scopes=scopes,
                           pair_id=pair, expires_at=now + config.REFRESH_TOKEN_TTL, label=label)
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=config.ACCESS_TOKEN_TTL,
                          scope=" ".join(scopes), refresh_token=refresh)

    def issue_personal_token(self, user_id: int, label: str = "personal") -> str:
        """Long-lived token for clients that cannot do OAuth (shown on the dashboard)."""
        token = security.new_token(32)
        self.db.save_token(token, kind="access", client_id=PERSONAL_CLIENT_ID, user_id=user_id,
                           scopes=[SCOPE], pair_id=security.new_token(12), expires_at=None, label=label)
        return token

    async def load_refresh_token(self, client: OAuthClientInformationFull,
                                 refresh_token: str) -> Optional[UserRefreshToken]:
        d = self.db.get_token(refresh_token, kind="refresh")
        if d is None or d["client_id"] != client.client_id:
            return None
        return UserRefreshToken(token=refresh_token, client_id=d["client_id"], scopes=d["scopes"],
                                expires_at=d["expires_at"], user_id=d["user_id"], pair_id=d["pair_id"])

    async def exchange_refresh_token(self, client: OAuthClientInformationFull,
                                     refresh_token: UserRefreshToken, scopes: list[str]) -> OAuthToken:
        user = self.db.get_user(refresh_token.user_id)
        if user is None:
            raise TokenError("invalid_grant", "account disconnected")
        self.db.revoke_pair(refresh_token.pair_id)   # rotate
        return self._issue(client.client_id or "", refresh_token.user_id, scopes or [SCOPE])

    async def load_access_token(self, token: str) -> Optional[UserAccessToken]:
        d = self.db.get_token(token, kind="access")
        if d is None:
            return None
        return UserAccessToken(token=token, client_id=d["client_id"], scopes=d["scopes"],
                               expires_at=d["expires_at"], user_id=d["user_id"])

    async def revoke_token(self, token) -> None:
        d = self.db.get_token(token.token)
        if d is not None:
            self.db.revoke_pair(d["pair_id"])


_provider: Optional[TelegramOAuthProvider] = None


def get_provider() -> TelegramOAuthProvider:
    global _provider
    if _provider is None:
        _provider = TelegramOAuthProvider()
    return _provider
