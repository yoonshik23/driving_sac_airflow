# dags/my_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from daily_push.data_push import make_push_data
from airflow.utils.dates import days_ago


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 24, 1, 0, 0),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_make_push_data',
    default_args=default_args,
    description='A simple data load DAG',
    schedule_interval=timedelta(days=1),
)

t1 = PythonOperator(
    task_id='make_push_data',
    python_callable = make_push_data,
    op_kwargs={
        'execution_date': '{{ ts }}'
    },
    dag=dag,
)
