All: step2

step1:
	python fetcher.py

step2: step1
	python data_parser.py

clean:
	rm -rf *.sgm *.csv