.PHONY: assets moe-fusion base-eval gt-lora mtp-lora moe-lora serve-reports

assets:
	bash scripts/keepedit/download_required_assets.sh
	python scripts/keepedit/check_required_assets.py

moe-fusion:
	bash scripts/keepedit/run_keepedit_moe_fusion.sh

base-eval:
	EXPERIMENT_NAME=qwen2511_base LORA_PATH=none bash scripts/keepedit/evaluate_qwen_edit_experiment.sh

gt-lora:
	bash scripts/keepedit/run_gt_lora_qwen_edit.sh

mtp-lora:
	bash scripts/keepedit/run_mtp_phasea.sh

moe-lora:
	bash scripts/keepedit/run_moe_teacher_lora.sh

serve-reports:
	python -m http.server 8899 --directory logs/keepedit
