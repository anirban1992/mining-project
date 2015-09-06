import requests


def fetch_sgm(base_url):
    """
        Fetches the sgm files and stores it in the dataset folder
        Keyword Arguments: base_url

        Raises:
            exception if folder does not exist or url does not exist
    """
    for i in range(0, 22):
        data_url = base_url.format(str(i).zfill(2))
        data = requests.get(data_url).content
        filename = 'data{}'.format(i)
        # let me know if you want to store the data as the same format 01.sgm or just 1.sgm,
        # also if there are any changes to document structure (as in extracting each body as a document etc)

        with open('dataset/{}.sgm'.format(filename), 'w') as f:
            f.write(data)


def main():
    fetch_sgm('http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0{}.sgm')

if __name__ == '__main__':
    main()

