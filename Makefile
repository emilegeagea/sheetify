.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
# run_preprocess:
# 	python -c 'from taxifare.interface.main import preprocess; preprocess()'
#
run_train:
	python -c 'from sheets.main import train; import tensorflow as tf; print(tf.config.list_physical_devices("GPU")); train(batch_size=64)'
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

download_and_train: reset_local_files download_gcs_files run_train

# download_unzip_train: reset_local_files download_gcs_files unzip_dl_to_data run_train

unzip_and_train: reset_local_files unzip_mounted_to_data run_train

symlink_and_train: reset_local_files symlink_bucket run_train

precompute_midi:
	python scripts/precompute_midi.py

precompute_cqt:
	python scripts/precompute_cqt.py

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/test_gcp_setup.py::TestGcpSetup::test_code_get_project


################### DATA SOURCES ACTIONS ################
# ML_DIR=./mlops

auth_gcs:
	-gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

list_gcs_files: auth_gcs
	gsutil iam get gs://${BUCKET_NAME}
	gsutil ls gs://${BUCKET_NAME}/data

download_gcs_files: auth_gcs
	-mkdir data
	-mkdir data/midis
	-mkdir data/mp3s
	gsutil cp gs://${BUCKET_NAME}/data/maestro-v3.0.0.json ./data
	gsutil -m cp -r gs://${BUCKET_NAME}/data/midis ./data
	gsutil -m cp -r gs://${BUCKET_NAME}/data/mp3s ./data

# Requires bucket to be mounted to the container
unzip_mounted_to_data:
	mkdir -p /tmp/training_data
	mkdir /tmp/training_data/midis
	mkdir /tmp/training_data/mp3s
	ls -Al /tmp/training_data/midis
	ls -Al /mnt/gcs
	unzip -q /mnt/gcs/maestro-v3.0.0-midi.zip -d /tmp/training_data/midis
	mv /tmp/training_data/midis/maestro-v3.0.0/* /tmp/training_data/midis
	unzip -q /mnt/gcs/maestro-v3.0.0-mp3.zip  -d /tmp/training_data/mp3s
	mv /tmp/training_data/mp3s/maestro-v3.0.0-mp3/* /tmp/training_data/mp3s
	ls -Al /tmp/training_data/midis
	ln -s /tmp/training_data ./data
	cp /mnt/gcs/data/maestro-v3.0.0.json ./data
	ls -Al /code/data

# Requires bucket to be mounted at /mnt/gcs
symlink_bucket:
	-rm -rf /code/data
	ln -s /mnt/gcs/data /code/data
	ls -Al /code/data
	ls -AlH /code/data

show_sources_all:
# 	-ls -laR ${ML_DIR}/data
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	-rm -rf ${ML_DIR}
	-mkdir -p ${ML_DIR}/data/
	-mkdir ${ML_DIR}/training_outputs
	-mkdir ${ML_DIR}/training_outputs/metrics
	-mkdir ${ML_DIR}/training_outputs/models
	-mkdir ${ML_DIR}/training_outputs/params

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
