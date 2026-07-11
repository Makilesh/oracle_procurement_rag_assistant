"""Shared dependencies: JWT verification and the per-user rate limiter."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import settings

ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Verify the bearer JWT and return its subject. 401 on any failure."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(subject)


def is_admin(user: str) -> bool:
    admins = {u.strip() for u in settings.admin_usernames.split(",") if u.strip()}
    return user in admins


async def get_admin_user(user: str = Depends(get_current_user)) -> str:
    """Admin gate on top of JWT auth. 403 — not 401 — because the caller IS
    authenticated, just not authorized for this endpoint."""
    if not is_admin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


def _rate_limit_key(request: Request) -> str:
    """Rate-limit per JWT subject when authenticated, else per client IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = jwt.decode(auth[7:], settings.jwt_secret, algorithms=[ALGORITHM])
            if payload.get("sub"):
                return f"user:{payload['sub']}"
        except JWTError:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
