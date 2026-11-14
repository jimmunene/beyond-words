"""Shared helpers: config loading and tiny JSON-over-UDP messaging between processes."""
import json
import pathlib
import socket

import yaml

ROOT = pathlib.Path(__file__).resolve().parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


class UDPSender:
    def __init__(self, port, host="127.0.0.1"):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, obj):
        self.sock.sendto(json.dumps(obj).encode(), self.addr)


class UDPReceiver:
    def __init__(self, port, host="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def poll(self):
        """Return all messages waiting right now (never blocks)."""
        msgs = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, OSError):
                break
            msgs.append(json.loads(data))
        return msgs
