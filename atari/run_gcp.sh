#!/bin/sh

JOBNAME = rl_gcp

gcloud ml-engine jobs submit training $JOBNAME --package-path=rl_gcp/trainer --module-name=trainer.a3c.py --region=europe-west4 --staging-bucket=gs://rl_gcp --config=$PWD/config.yaml --runtime-version=1.10 --
