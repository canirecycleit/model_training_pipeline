FROM python:3.8-slim-buster AS builder

RUN apt-get update

WORKDIR /app

ENTRYPOINT ["/bin/bash", "shell_scripts/supervisord_entrypoint.sh"]
CMD ["-c", "/etc/supervisor/supervisord.conf"]