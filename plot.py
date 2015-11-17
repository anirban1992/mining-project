"""

Output of my_data_hasher.py

/usr/bin/python /Users/priyanka/Desktop/Data_Mining/Code/mining-project/my_data_hasher.py
Shingling articles...

Shingling 19043 docs took 3.31 sec.

Average shingles per doc: 137.83

Calculating Jaccard Similarities...
  (0 / 19043)
  (1000 / 19043)
  (2000 / 19043)
  (3000 / 19043)
  (4000 / 19043)
  (5000 / 19043)
  (6000 / 19043)
  (7000 / 19043)
  (8000 / 19043)
  (9000 / 19043)
  (10000 / 19043)
  (11000 / 19043)
  (12000 / 19043)
  (13000 / 19043)
  (14000 / 19043)
  (15000 / 19043)
  (16000 / 19043)
  (17000 / 19043)
  (18000 / 19043)
  (19000 / 19043)

Calculating all Jaccard Similarities took 15022.51sec

Generating random hash functions...

Generating MinHash signatures for all documents...

Generating MinHash signatures took 63.38sec

Comparing all signatures...

Comparing MinHash signatures took 737.09sec
For k= 16

Sum is : 87813.9705493

SSE is: 4.61135170662

Generating random hash functions...

Generating MinHash signatures for all documents...

Generating MinHash signatures took 99.72sec

Comparing all signatures...

Comparing MinHash signatures took 12839.62sec
For k= 32

Sum is : 67206.5986154

SSE is: 3.52920225885

Generating random hash functions...

Generating MinHash signatures for all documents...

Generating MinHash signatures took 115.70sec

Comparing all signatures...

Comparing MinHash signatures took 4433.85sec
For k= 64

Sum is : 14181.1968037

SSE is: 0.744693420349

Generating random hash functions...

Generating MinHash signatures for all documents...

Generating MinHash signatures took 67.07sec

Comparing all signatures...

Comparing MinHash signatures took 2751.41sec
For k= 128

Sum is : 8127.06311283

SSE is: 0.426774306193

Generating random hash functions...

Generating MinHash signatures for all documents...

Generating MinHash signatures took 153.00sec

Comparing all signatures...


Comparing MinHash signatures took 6269.49sec
For k= 256

Sum is : 4624.31923291

SSE is: 0.242835647372

"""

import matplotlib.pyplot as plt

hash_range = [16,32,64,128,256]
sse = [4.61135170662,3.52920225885, 0.744693420349,  0.426774306193 , 0.242835647372]

efficacy = [737.09 , 12839.62 , 4433.85 , 2751.41 , 6269.49 ]

plt.plot(hash_range, sse, 'xb-')
plt.axis([5, 280, min(sse) - 0.2, max(sse) + 1])
plt.show()

plt.plot(hash_range, efficacy, 'xr-')
plt.axis([5, 280, min(efficacy) - 100, max(efficacy) + 50])
plt.show()
