#!/bin/bash

# Function to check if user exists
user_exists() {
    id "$1" &>/dev/null
}

# Function to check if group exists
group_exists() {
    getent group "$1" &>/dev/null
}

# Function to create airflow user and group if not exists
create_airflow_user_and_group() {
    if ! group_exists "airflow"; then
        sudo groupadd -r airflow
        echo "Created airflow group"
    else
        echo "Airflow group already exists"
    fi

    if ! user_exists "airflow"; then
        sudo useradd -r -g airflow -d $AIRFLOW_HOME airflow
        echo "Created airflow user"
    else
        echo "Airflow user already exists"
    fi
}

# Create airflow user and group if necessary
create_airflow_user_and_group
  
# Initialize Airflow database
airflow db init

# Update URLs in airflow.cfg
sed -i 's/localhost/163.152.172.163/g' $AIRFLOW_HOME/airflow.cfg

# Add or update logging_config_class in airflow.cfg
if grep -q '^\s*logging_config_class\s*=' $AIRFLOW_HOME/airflow.cfg; then
    sed -i 's|^\s*logging_config_class\s*=.*|logging_config_class = airflow.config_templates.airflow_local_settings.DEFAULT_LOGGING_CONFIG|g' $AIRFLOW_HOME/airflow.cfg
else
    sed -i '/\[logging\]/a logging_config_class = airflow.config_templates.airflow_local_settings.DEFAULT_LOGGING_CONFIG' $AIRFLOW_HOME/airflow.cfg
fi

sudo mkdir -p $AIRFLOW_HOME/logs
sudo chown -R airflow:airflow $AIRFLOW_HOME/logs
sudo chmod -R 775 $AIRFLOW_HOME/logs

# Set the FLASK_APP environment variable
export FLASK_APP=airflow.www.app:create_app

# Function to create admin user using flask fab command
create_admin_user() {
    flask fab create-admin \
        --username admin \
        --firstname Admin \
        --lastname User \
        --email admin@example.com \
        --password admin
}

# Check if the admin user already exists using flask fab command
EXISTING_USER=$(flask fab list-users | grep -w admin)
if [ -z "$EXISTING_USER" ]; then
    echo "Creating admin user"
    create_admin_user
else
    echo "Admin user already exists"
fi

# Start the original entrypoint command
exec "$@"

