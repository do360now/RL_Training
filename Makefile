GAME = Breakout-v4


train:
	python3 train.py --env $(GAME) --timesteps 51200 2>&1 | tee -a logs/$(GAME).log

train-render:
	python3 train.py --env $(GAME) --timesteps 2048 --render 2>&1 | tee -a logs/$(GAME).log

train-pong:
	python3 train.py --env Pong-v4 --timesteps 51200 > logs/pong.log

view:
	tensorboard --logdir logs

disk-usage:
	du -sh * | sort -h

