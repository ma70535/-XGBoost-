import requests

def download_dataset(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

if __name__ == '__main__':
    # 以Kaggle公开数据为例
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
    filename = 'creditcard.csv'
    download_dataset(url, filename)