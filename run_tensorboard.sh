ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9

tensorboard --port 6007 --logdir outputs/lightning_logs