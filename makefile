All: step2

step1:
	python fetcher.py

step2: step1
	python data_parser.py

step3: step2
    python kmeans.py

clean:
	rm -rf *.sgm *.csv