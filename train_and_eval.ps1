$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Python = "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $Python -m pip install -r requirements-prod.txt
& $Python train_export_domain_classifier.py --data canteen_test_data_300.json
& $Python production_gateway.py --data canteen_test_data_300.json
