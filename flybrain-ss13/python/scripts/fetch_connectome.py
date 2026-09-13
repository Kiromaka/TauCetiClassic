#!/usr/bin/env python3
"""Скачать выгрузку коннектома FlyWire (FAFB, снапшот 783).

Файлы лежат в публичном бакете Codex:
    https://storage.googleapis.com/flywire-data/codex/data/fafb/783/

Данные FlyWire распространяются по CC BY-NC-SA 4.0 — некоммерческое
использование со ссылкой. Цитировать нужно:
    Dorkenwald et al., Nature 2024 (реконструкция)
    Schlegel et al.,  Nature 2024 (аннотации и типы клеток)
    Shiu et al.,      Nature 2024 (сама LIF-модель)
"""
import argparse
import os
import sys
import urllib.request

BASE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/{ver}"
# В снапшотах после 630 колонки cell_type в classification.csv больше нет:
# типы клеток уехали в consolidated_cell_types.csv. Без них не найти ни
# нисходящие нейроны, ни вкусовые популяции, поэтому файл обязательный.
FILES = {
    "connections.csv.gz": True,
    "classification.csv.gz": True,
    "consolidated_cell_types.csv.gz": True,
    "coordinates.csv.gz": False,         # для ретинотопии
    "neurons.csv.gz": False,             # запасной источник медиатора
    "cell_types.csv.gz": False,
    "visual_neuron_types.csv.gz": False,
    "labels.csv.gz": False,
}


def fetch(url: str, dest: str) -> None:
    print(f"  {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r    {done/1e6:7.1f} / {total/1e6:.1f} МБ  {pct:3d}%",
                      end="", flush=True)
    print()
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="783")
    ap.add_argument("--out", default="data")
    ap.add_argument("--all", action="store_true",
                    help="качать и необязательные файлы")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    base = BASE.format(ver=a.version)
    ok = True
    for name, required in FILES.items():
        if not required and not a.all:
            continue
        dest = os.path.join(a.out, name)
        if os.path.exists(dest):
            print(f"уже есть: {dest}")
            continue
        try:
            fetch(f"{base}/{name}", dest)
        except Exception as exc:  # noqa: BLE001
            msg = f"не смог скачать {name}: {exc}"
            if required:
                print("ОШИБКА: " + msg, file=sys.stderr)
                ok = False
            else:
                print("пропускаю (необязательный): " + msg)
    if ok:
        print("\nГотово. Дальше:  python3 scripts/build_connectome.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
