.PHONY: all

# Source Files
ENVIRONMENT = environment
EXPERIMENT = experiment
AGENT = hrlAgent
SRC_PATH = ./src

# Path to RL-Glue to find RL_interface and related files.
PYTHON_CODEC_PATH = ../external/python-codec/src/
PARAMETER_HOLDER_PATH = ../external/15-rl-competition-2009/system/includes
LOADER_HELPER_PATH = ../external/15-rl-competition-2009/trainers/consoleTrainerPython/src/

all: 3room_environment 3room_agent 3room_experiment

3room_environment:
	echo -e "#! /bin/bash\n  PYTHONPATH=$(PYTHON_CODEC_PATH):$(PARAMETER_HOLDER_PATH):$(LOADER_HELPER_PATH):$(SRC_PATH) python $(SRC_PATH)/$(ENVIRONMENT).py " | cat > 3room_environment
	chmod u+x 3room_environment

3room_agent:
	echo "#! /bin/bash\n  PYTHONPATH=$(PYTHON_CODEC_PATH) python $(SRC_PATH)/$(AGENT).py " | cat > 3room_agent
	chmod u+x 3room_agent

3room_experiment:
	echo "#! /bin/bash\n  PYTHONPATH=$(PYTHON_CODEC_PATH) python $(SRC_PATH)/$(EXPERIMENT).py " | cat > 3room_experiment
	chmod u+x 3room_experiment

clean:
	rm -f 3room_environment 3room_agent 3room_experiment *.pyc $(SRC_PATH)/*.pyc
