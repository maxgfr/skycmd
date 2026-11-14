# s6-secrets v2 (docs/s6-secrets2-prereg.md, section 6): Betterleaks 1.10.1, a third-party
# executable, in a container that is started without a network and sees only the folders of
# the split it scans, read-only. The binary comes from the release tarball pinned in
# scripts/secrets2/pins.json (setup_tools.py checks its sha256 and extracts it here).
ARG ALPINE
FROM ${ALPINE}
COPY betterleaks /usr/local/bin/betterleaks
USER 65534:65534
WORKDIR /in
ENTRYPOINT ["/usr/local/bin/betterleaks"]
