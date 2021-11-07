FROM python:3.8-slim-buster AS builder

RUN apt-get update && apt-get -y install cron && apt-get -y install git

# Add crontab file in the cron directory
ADD crontab /etc/cron.d/simple-cron

# Add shell script and grant execution rights
ADD shell_script/script.sh /script.sh
RUN chmod +x /script.sh

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/simple-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log


WORKDIR /app

COPY . /app

# Update PIP & install requirements
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the command on container startup
CMD cron && tail -f /var/log/cron.log