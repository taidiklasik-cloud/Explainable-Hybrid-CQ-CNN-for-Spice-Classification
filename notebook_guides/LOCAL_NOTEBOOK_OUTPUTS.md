# Penyimpanan Output Notebook Lokal

Output notebook tidak dimasukkan ke PostgreSQL.

## Prinsip

- CSV, JSON, PNG/JPG/SVG, HTML, dan notebook executed disimpan di laptop/worker.
- PostgreSQL lokal hanya menyimpan orchestration training dan metadata checkpoint.
- Checkpoint `.pt` penting disimpan di Google Drive via rclone.
- Worker menyimpan `latest.pt` lokal untuk self-recovery cepat.

## Struktur folder lokal yang disarankan

```text
outputs/
├── 01_eda_datasetawal/
│   ├── csv/
│   ├── json/
│   ├── figures/
│   └── notebook_runs/
├── 02_eda_fiturgambar/
│   ├── csv/
│   ├── json/
│   ├── figures/
│   └── notebook_runs/
├── 03_posprocessing_eda_outlierdetection/
│   ├── csv/
│   ├── json/
│   ├── figures/
│   └── notebook_runs/
├── 04_curriculum_stage/
│   ├── manifests/
│   ├── csv/
│   ├── json/
│   └── notebook_runs/
└── local_artifact_manifest.csv
```

## Kolom manifest lokal opsional

```text
notebook_name, output_group, file_type, file_path, sha256, created_at, description
```

Manifest ini hanya untuk audit lokal dan lampiran skripsi.
