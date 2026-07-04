import socket
import struct
import json
import zlib
import time

HOST = "127.0.0.1"
PORT = 9000

PACKET_HEADER_SIZE = 36
MAGIC = 0x424C4144

TYPE_IMAGE_JPEG = 1
TYPE_META_JSON = 2
TYPE_HEARTBEAT_PING = 3
TYPE_HEARTBEAT_PONG = 4
TYPE_COMMAND = 5
TYPE_ERROR_TEXT = 6


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("server disconnected")
        data += chunk
    return data


def make_packet(packet_type, sequence, payload=b""):
    magic = MAGIC
    version = 1
    flags = 0
    timestamp_ms = int(time.time() * 1000)
    payload_size = len(payload)
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF

    header = struct.pack(
        ">IHHIQQII",
        magic,
        version,
        packet_type,
        flags,
        sequence,
        timestamp_ms,
        payload_size,
        crc32
    )

    return header + payload


def send_command(sock, sequence, text):
    payload = text.encode("utf-8")
    sock.sendall(make_packet(TYPE_COMMAND, sequence, payload))


def send_pong(sock, sequence):
    sock.sendall(make_packet(TYPE_HEARTBEAT_PONG, sequence, b""))


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    print("connected")

    seq = 1
    send_command(sock, seq, "start")
    seq += 1

    image_count = 0

    while True:
        header = recv_exact(sock, PACKET_HEADER_SIZE)

        magic, version, packet_type, flags, sequence, timestamp_ms, payload_size, crc = struct.unpack(
            ">IHHIQQII",
            header
        )

        if magic != MAGIC:
            raise RuntimeError(f"bad magic: {hex(magic)}")

        payload = recv_exact(sock, payload_size) if payload_size > 0 else b""

        real_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if real_crc != crc:
            raise RuntimeError("crc check failed")

        if packet_type == TYPE_IMAGE_JPEG:
            image_count += 1
            filename = f"recv_{image_count:04d}.jpg"
            with open(filename, "wb") as f:
                f.write(payload)
            print(f"recv image: {filename}, size={len(payload)}")

        elif packet_type == TYPE_META_JSON:
            text = payload.decode("utf-8")
            print("recv json:", text)

        elif packet_type == TYPE_HEARTBEAT_PING:
            print("recv heartbeat ping")
            send_pong(sock, seq)
            seq += 1

        elif packet_type == TYPE_ERROR_TEXT:
            print("recv error:", payload.decode("utf-8"))

        else:
            print("recv packet type:", packet_type, "size:", payload_size)