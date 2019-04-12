#!/bin/sh

JOBNAME=rl_gcp_$(date -u +%y%m%d_%H%M%S)

gcloud ml-engine jobs submit training $JOBNAME \
    --package-path=rl_gcp/trainer \
    --module-name=trainer.a3c \
    --region=europe-west4 \
    --staging-bucket=gs://rl_gcp \
    --config=config.yaml \
    --runtime-version=1.10 \
    --python-version=3.5
