#!/bin/sh
WORKER_TARGET="ml:3"
PS_TARGET="ml:4"
HELPER="ml:5"
WORKERS=8
PS_SERVERS=2
HOME_DIR="/home/peachball/D/git/ML-Papers/pyml"
LOGDIR="$HOME_DIR/tflogs/lstmcartpole/"
ENV="CartPole-v0"

tmux kill-window -t $HELPER
tmux kill-window -t $WORKER_TARGET
tmux kill-window -t $PS_TARGET

sleep 1
rm -r $LOGDIR

tmux new-window -t $WORKER_TARGET
for i in $(seq 0 $(($WORKERS - 2)))
do
	tmux split-window -t "$WORKER_TARGET.$i"
	tmux select-layout -t $WORKER_TARGET tile
done
for i in $(seq 0 $(($WORKERS - 1)))
do
	tmux send-keys -t "$WORKER_TARGET.$i" "cd $HOME_DIR" ENTER
	tmux send-keys -t "$WORKER_TARGET.$i" "CUDA_VISIBLE_DEVICES= \
python3 AsyncRL.py \
--task $i --cluster_file sample-hosts.txt --type worker --env $ENV --logdir \
$LOGDIR" ENTER
done

tmux new-window -t $PS_TARGET
for i in $(seq 0 $(($PS_SERVERS - 2)))
do
	tmux split-window -t "$PS_TARGET.$i"
	tmux select-layout -t $PS_TARGET tile
done

for i in $(seq 0 $(($PS_SERVERS- 1)))
do
	tmux send-keys -t "$PS_TARGET.$i" "cd $HOME_DIR" ENTER
	tmux send-keys -t "$PS_TARGET.$i" "CUDA_VISIBLE_DEVICES= python3 AsyncRL.py \
--task $i --cluster_file sample-hosts.txt --type ps --env $ENV" ENTER
done

tmux new-window -t $HELPER
tmux send-keys -t $HELPER "cd $HOME_DIR" ENTER
tmux send-keys -t $HELPER "tensorboard --logdir $LOGDIR" ENTER
