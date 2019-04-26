#!/bin/sh

gcloud ai-platform local train \
    --module-name=trainer.a3c \
    --package-path=$PWD/rl_gcp/trainer \
    --distributed \
    --parameter-server-count=1 \
    --worker-count=3 \
    -- \
    --max_step=2000
