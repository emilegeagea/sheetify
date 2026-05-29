.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
# run_preprocess:
# 	python -c 'from taxifare.interface.main import preprocess; preprocess()'
#
run_train:
	python -c 'from sheets.main import train; train()'
#
# run_pred:
# 	python -c 'from taxifare.interface.main import pred; pred()'
#
# run_evaluate:
# 	python -c 'from taxifare.interface.main import evaluate; evaluate()'
#
# run_all: run_preprocess run_train run_pred run_evaluate
#
# run_workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow


##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/test_gcp_setup.py::TestGcpSetup::test_code_get_project


################### DATA SOURCES ACTIONS ################
ML_DIR=~/.lewagon/mlops/sheetify

show_sources_all:
	-ls -laR ${ML_DIR}/data
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ${ML_DIR}/data/
	mkdir ${ML_DIR}/training_outputs
	mkdir ${ML_DIR}/training_outputs/metrics
	mkdir ${ML_DIR}/training_outputs/models
	mkdir ${ML_DIR}/training_outputs/params

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_gcs_files


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -rf **/__pycache__ **/*.pyc
	@rm -rf **/build **/dist
	@rm -rf proj-*.dist-info
	@rm -rf proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
