All: step6

step1:
	fetcher.py

step2: step1
	python data_parser.py

clean:
	rm -rf *.sgm *.csv hello