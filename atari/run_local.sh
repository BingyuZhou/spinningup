#!/bin/sh

gcloud ml-engine local train \
    --module-name=trainer.a3c \
    --package-path=$PWD/rl_gcp/trainer \
    --distributed \
    --parameter-server-count=1 \
    --worker-count=4 \
    -- \
    --max_step=2000
