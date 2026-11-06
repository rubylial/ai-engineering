"""Punto de entrada: `python -m intelligence_system` (host y puerto desde Settings)."""

from __future__ import annotations

import uvicorn

from intelligence_system.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "intelligence_system.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # el logging lo configura create_app (JSON)
        access_log=False,  # el access log lo emite el middleware con request_id
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
