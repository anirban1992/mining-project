All: step2

step1:
	python fetcher.py

step2: step1
	python my_data_hasher.py
clean:
	rm -rf *.sgm *.csv