"""Envio de la respuesta final a watbot_v2.

El VPS no habla con Meta: watbot_v2 recibe la respuesta en /callback y es quien
la envia a WhatsApp.
"""

import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

TIMEOUT = 10.0
MAX_INTENTOS = 3


async def enviar_respuesta(settings: Settings, phone: str, response: str) -> bool:
    """POST {phone, response} al callback de watbot_v2 con Bearer token."""
    headers = {
        "Authorization": f"Bearer {settings.WATBOT_CALLBACK_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"phone": phone, "response": response}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for intento in range(MAX_INTENTOS):
            try:
                r = await client.post(settings.WATBOT_CALLBACK_URL, json=payload, headers=headers)
                if r.status_code < 400:
                    return True
                logger.error("Callback respondio %d", r.status_code)
                # 4xx no se reintenta: el request es invalido y no va a cambiar.
                if r.status_code < 500:
                    return False
            except httpx.HTTPError as exc:
                logger.error("Fallo el callback (intento %d): %s", intento + 1, type(exc).__name__)

    return False
