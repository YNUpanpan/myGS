#!/usr/bin/env python
"""Run pip with DNS resolution forced to IPv4.

Some hosts prefer IPv6 addresses even when IPv6 routing to package CDNs is not
usable. This wrapper keeps pip's normal behavior but restricts getaddrinfo to
AF_INET so PyTorch wheels can be fetched over IPv4.
"""

import runpy
import socket
import sys


def main() -> None:
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_getaddrinfo
    sys.argv = ["pip"] + sys.argv[1:]
    runpy.run_module("pip", run_name="__main__")


if __name__ == "__main__":
    main()
