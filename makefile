train: 
	rm -rf logdir/*
	python train.py
traincc:
	rm -rf logdir/*
	python train0616.py
clean:
	rm -rf logdir/* *.pyc
