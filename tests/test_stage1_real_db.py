import os

from _path_setup import PROJECT_ROOT, configure_paths

configure_paths()

os.environ.setdefault("DATASET_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("CURRICULUM_OUTPUTS_ROOT", str(PROJECT_ROOT / "curriculum_outputs"))
os.environ.setdefault("REQUIRE_CHECKPOINT_UPLOAD", "false")

from dotenv import load_dotenv
load_dotenv()
dsn = os.environ.get('ORCHESTRATION_DB_DSN')

from patched_program_files.postgres_orchestration_db import PostgresOrchestrationDb, PostgresOrchestrationDbConfig
cfg = PostgresOrchestrationDbConfig(dsn=dsn)
db = PostgresOrchestrationDb(cfg)

from test_stage1_sanity_smoke import build_mock_task, build_trial_params
from patched_program_files.worker_task_template import real_train_one_task

task = build_mock_task()
trial_params = build_trial_params()

def heartbeat_callback():
    print("[test] dummy heartbeat callback")
    return True

print("Running real_train_one_task with REAL DB connection...")
try:
    result = real_train_one_task(
        task=task,
        db=db,
        worker_uid="pc_01",
        trial_params=trial_params,
        heartbeat_callback=heartbeat_callback
    )
    print("Finished!")
    print(result)
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("[test] Cleaning up test task data from DB...")
    db.delete_task_data(task["task_id"])
    print("[test] Cleanup complete.")
