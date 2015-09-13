import requests
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

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
        filename = 'data{}'.format(str(i).zfill(2))
        with open('dataset/{}.sgm'.format(filename), 'w') as f:
            f.write(data)


def main():
    fetch_sgm('http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0{}.sgm')

if __name__ == '__main__':
    main()

