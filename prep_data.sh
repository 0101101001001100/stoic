#!/bin/bash

nohup python ./scanner/prep.py --in_dir /data/databases/stoic/data/mha/ --out_dir /data/databases/stoic/data/uni/ --workers 8 > prep_data.log &