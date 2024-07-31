# Use the official Airflow image
FROM apache/airflow:2.5.0-python3.9

# Switch to the root user to install packages
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip vim sudo && \
    echo "airflow ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# Switch back to the airflow user
USER airflow

# Set the environment variable for AIRFLOW_HOME
ENV AIRFLOW_HOME=/opt/airflow
WORKDIR /opt/airflow

# Copy requirements.txt and install packages
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --user -r /requirements.txt

ENV PATH=/home/airflow/.local/bin:$PATH


# Copy the DAGs, logs, and plugins
COPY dags $AIRFLOW_HOME/dags
COPY logs $AIRFLOW_HOME/logs
COPY plugins $AIRFLOW_HOME/plugins
USER root

# Copy the script to update airflow.cfg
COPY update_airflow_cfg.sh /update_airflow_cfg.sh
RUN chmod +x /update_airflow_cfg.sh

USER airflow
# Run the script to update airflow.cfg
ENTRYPOINT ["/update_airflow_cfg.sh"]
