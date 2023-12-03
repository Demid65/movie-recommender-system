from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

DATASET_LINK = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
EXTRACT_FOLDER = 'data/raw/'
PROCESSED_PATH = 'benchmark/data/dataset.csv'

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python .\benchmark\download_dataset.py',
                    description='downloads the dataset for movie recommender, preprocesses it and saves it for further use.',
                    epilog='https://github.com/Demid65/movie-recommender-system')
                    
parser.add_argument('--link', type=str, metavar='DATASET_LINK', dest='DATASET_LINK',
                    help=f'Link to the dataset. Defaults to {DATASET_LINK}', default=DATASET_LINK)

parser.add_argument('--extract_to', type=str, metavar='EXTRACT_FOLDER', dest='EXTRACT_FOLDER',
                    help=f'Folder where dataset is extracted to. Defaults to {EXTRACT_FOLDER}', default=EXTRACT_FOLDER)
                            
parser.add_argument('--save_to', type=str, metavar='OUTPUT_FILE', dest='PROCESSED_PATH',
                    help=f'Path where processed dataset is saved. Defaults to {PROCESSED_PATH}', default=PROCESSED_PATH) 

args = parser.parse_args()

DATASET_LINK = args.DATASET_LINK
EXTRACT_FOLDER = args.EXTRACT_FOLDER
PROCESSED_PATH = args.PROCESSED_PATH


# Download and unzip the dataset
def download_and_unzip(url, extract_to='.'): # https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


print(f'downloading from {DATASET_LINK}')
download_and_unzip(DATASET_LINK, extract_to=EXTRACT_FOLDER)

print(f'reading dataset from {EXTRACT_FOLDER}')

datapath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.data')
data = pd.read_csv(datapath, sep='\t', header=None)

genrespath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.genre')
genres = pd.read_csv(genrespath, sep='|', header=None)

userspath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.user')
users = pd.read_csv(userspath, sep='|', header=None)

userspath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.user')
users = pd.read_csv(userspath, sep='|', header=None)

itemspath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.item')
items = pd.read_csv(itemspath, sep='|', header=None, encoding='ANSI').drop(columns=[3])

occpath = os.path.join(EXTRACT_FOLDER, 'ml-100k/u.occupation')
occupations = pd.read_csv(occpath, sep='|', header=None)
o_dict = {v: k for k, v in occupations.to_dict()[0].items()}

def vectorize(user, movie, rating):
    o_vec = [0] * len(o_dict)
    o_vec[o_dict[user[3]]]  = 1
    if movie[1] == 'unknown': # strange edge case, better get rid of those
        return -1
    return [user[1], int(user[2] == 'M')]+o_vec+[int(movie[2].split('-')[2])]+list(movie.drop([0, 1, 2, 4]))+[rating]

print('processing')

d = {}
for i, row in tqdm(data.iterrows(), total=len(data)):
    
    user = users.iloc[row[0]-1]
    movie = items.iloc[row[1]-1]
    rating = row[2]
    vec = vectorize(user, movie, rating)
    if vec == -1:
        continue
        
    d.update({i:vec})

dataset = pd.DataFrame.from_dict(d, orient='index')

print(f'saving into {PROCESSED_PATH}')

dataset.to_csv(PROCESSED_PATH)

print('done')