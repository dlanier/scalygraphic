SCRIPT = ../src/scalygraphic.py
RUN_DIR = ./run_dir
RUN_FILES = ../data/run_files
RESULTS_DIR = $(RUN_DIR)/results

RMR = rm -rf
MKDIR_P = mkdir -p

env_setup: create_run_dir copy_run_files

create_run_dir:
	$(MKDIR_P) $(RESULTS_DIR)

copy_run_files:
	cp $(RUN_FILES)/*.yml $(RUN_DIR)

run_scaled_images_dataset:
	python3 $(SCRIPT) -run_directory $(RUN_DIR) -run_file create_scaled_image_set.yml

