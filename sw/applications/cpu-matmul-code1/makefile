# Copyright 2023 EPFL and Politecnico di Torino.
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# File: makefile
# Author: Michele Caon
# Date: 24/07/2023
# Description: Makefile for multi-accels matrix multiplication app

#############################
# ----- CONFIGURATION ----- #
#############################

KERNEL_PARAMS 		?= int32 --row_a 8 --col_a 8 --col_b 256

#####################
# ----- RULES ----- #
#####################

# Default alias
.PHONY: all
all: data.h

# Generate data.h
data.h: .force
	@echo "## Generating data.h..."
	PYTHONPATH=../../nmc python3 datagen.py $(KERNEL_PARAMS)

# Dummy rule to force recompilation (because of parameters)
.force:

# Clean
.PHONY: clean
clean:
	$(RM) data.h carus_data.h caesar_data.h caesar_commands.h caesar_instructions*
