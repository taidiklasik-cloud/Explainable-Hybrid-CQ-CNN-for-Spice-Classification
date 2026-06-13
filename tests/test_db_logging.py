import os
from dotenv import load_dotenv

from _path_setup import configure_paths

configure_paths()

load_dotenv()
dsn = os.environ.get('ORCHESTRATION_DB_DSN')
print("DSN:", dsn)
from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
cfg = PostgresOrchestrationDbConfig(dsn=dsn)
db = PostgresOrchestrationDb(cfg)
print("Connection test:", db.test_connection())
print("Stage info for stage 1 classical:", db.get_stage_info(1, "classical"))
