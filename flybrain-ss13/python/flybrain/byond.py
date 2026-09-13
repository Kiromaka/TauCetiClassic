"""Клиент протокола world/Topic BYOND.

Обратный канал Python -> сервер. Нужен только для внеполосных команд
(спавн мухи, перезагрузка мозга, отладка): основной цикл идёт быстрее —
DM сам стучится по HTTP и получает моторную команду в теле ответа.

Формат запроса:
    \\x00\\x83  <len:uint16 BE>  \\x00\\x00\\x00\\x00\\x00  <query>  \\x00
    len = len(query) + 6

Формат ответа:
    \\x00\\x83  <len:uint16 BE>  <type>  <payload>
    type 0x2a -> float32 LE, type 0x06 -> строка, завершённая нулём
"""
from __future__ import annotations

import socket
import struct
import urllib.parse
from typing import Optional, Union


class TopicError(RuntimeError):
    pass


def topic(host: str, port: int, query: str, timeout: float = 2.0
          ) -> Union[str, float, None]:
    """Отправить topic-запрос. query — без ведущего '?'."""
    if not query.startswith("?"):
        query = "?" + query
    payload = b"\x00\x00\x00\x00\x00" + query.encode("utf-8") + b"\x00"
    packet = b"\x00\x83" + struct.pack(">H", len(payload)) + payload

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(packet)
        head = _recv_exact(sock, 4)
        if head[:2] != b"\x00\x83":
            raise TopicError(f"неожиданный ответ: {head!r}")
        length = struct.unpack(">H", head[2:4])[0]
        if length == 0:
            return None
        body = _recv_exact(sock, length)

    kind = body[0]
    if kind == 0x2A:                       # float
        return struct.unpack("<f", body[1:5])[0]
    if kind == 0x06:                       # строка
        return body[1:].split(b"\x00", 1)[0].decode("utf-8", "replace")
    return None


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise TopicError("соединение закрыто раньше времени")
        buf += chunk
    return buf


def send_command(host: str, port: int, secret: str, **params) -> Union[str, float, None]:
    """Отправить команду модулю flybrain на сервере."""
    q = {"flybrain": "", "secret": secret}
    q.update({k: v for k, v in params.items() if v is not None})
    return topic(host, port, urllib.parse.urlencode(q, safe=""))


def params2list(s: str) -> dict:
    """Разобрать строку в формате BYOND list2params."""
    out = {}
    for part in s.split("&"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
    return out
