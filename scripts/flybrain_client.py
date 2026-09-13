#!/usr/bin/env python3
"""Мост для медленного (файлового) транспорта.

DM пишет пакет в файл и зовёт этот скрипт через world.ext_python().
Скрипт просто перекладывает пакет в HTTP-запрос к уже работающему демону
и кладёт ответ в выходной файл.

Важно: демон при этом остаётся один и тот же процесс, коннектом не
перезагружается. Здесь порождается только тонкий клиент — иначе каждый
тик стоил бы десятков секунд на загрузку 15 миллионов связей.

Зависимостей нет, только стандартная библиотека: скрипт должен стартовать
за десятки миллисекунд.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:5665")
    ap.add_argument("--secret", default="")
    ap.add_argument("--timeout", type=float, default=4.0)
    a = ap.parse_args()

    try:
        with open(a.inp, "rb") as fh:
            body = fh.read()
    except OSError as exc:
        sys.stderr.write(f"не прочитал {a.inp}: {exc}\n")
        return 1

    req = urllib.request.Request(
        a.url.rstrip("/") + "/tick", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "X-Flybrain-Secret": a.secret})
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        payload = json.dumps({"error": f"HTTP {exc.code}",
                              "body": exc.read()[:200].decode("utf-8", "replace")}
                             ).encode()
    except Exception as exc:  # noqa: BLE001
        payload = json.dumps({"error": str(exc)}).encode()

    tmp = a.out + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, a.out)          # атомарно: DM не увидит половину файла
    try:
        os.unlink(a.inp)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
