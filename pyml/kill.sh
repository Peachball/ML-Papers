#!/usr/bin/env sh
WORKER_TARGET="ml:3"
PS_TARGET="ml:4"

tmux kill-window -t $WORKER_TARGET
tmux kill-window -t $PS_TARGET
