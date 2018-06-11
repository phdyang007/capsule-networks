train: 
	rm -rf logdir/*
	python train.py

clean:
	rm -rf logdir/* *.pyc