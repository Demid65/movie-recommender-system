from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch
import pandas as pd
import os
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

DATASET_PATH = 'benchmark/data/dataset.csv'
MODEL_PATH = 'models/model.pt'
BATCH_SIZE = 512

# parse the command line arguments
parser = argparse.ArgumentParser(
                    prog='python .\benchmark\evaluate.py',
                    description='evaluates the model using Mean Average Error and Mean Squared Error.',
                    epilog='https://github.com/Demid65/movie-recommender-system')
                    
parser.add_argument('--model', type=str, metavar='MODEL_PATH', dest='MODEL_PATH',
                    help=f'Path to the model checkpoint. Defaults to {MODEL_PATH}', default=MODEL_PATH)

parser.add_argument('--dataset', type=str, metavar='DATASET_PATH', dest='DATASET_PATH',
                    help=f'Path to the processed dataset. Defaults to {DATASET_PATH}', default=DATASET_PATH)
                            
parser.add_argument('--batch_size', type=int, metavar='BATCH_SIZE', dest='BATCH_SIZE',
                    help=f'Batch size for prediction. Defaults to {BATCH_SIZE}', default=BATCH_SIZE) 

args = parser.parse_args()

DATASET_PATH = args.DATASET_PATH
MODEL_PATH = args.MODEL_PATH
BATCH_SIZE = args.BATCH_SIZE

torch.manual_seed(1337)

print(f'loading dataset from {DATASET_PATH}')

data = pd.read_csv(DATASET_PATH, index_col=0)

def norm_year(year):
    return year - 1900
    
data['23'] = data['23'].apply(norm_year)
data['23']
X = data.drop(columns=['43'])
y = data['43']

Xt = torch.tensor(X.values, dtype=torch.float)
yt = torch.tensor(y, dtype=torch.float)

dataset = TensorDataset(Xt, yt)

batch_size = 512
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#model
class RatingModel(nn.Module):

    def __init__(self, input_dim):
        super(RatingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

#prediction fucntion
def predict(model, dataloader, device="cpu"):
    
        with torch.no_grad():
            results = None
            targets = None
            model.eval()  # evaluation mode
            loop = tqdm(enumerate(dataloader, 0), total=len(dataloader))
            for i, data in loop:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                labels = torch.squeeze(labels)
                outputs = torch.squeeze(outputs)

                if results is None:
                    results = outputs
                    targets = labels
                else:
                    results = torch.cat((results, outputs))
                    targets = torch.cat((targets, labels))
                
        return results.cpu(), targets.cpu()

# load the model and get the predictions
print(f'loading checkpoint from {MODEL_PATH}')

model = RatingModel(input_dim=43)
ckpt = torch.load(MODEL_PATH)
model.load_state_dict(ckpt)

device = 'cuda' if torch.cuda.is_available else 'cpu'
model = model.to(device)

print(f'running prediction')
preds, labels = predict(model, dataloader, device)

MAEScore = nn.functional.l1_loss(preds, labels).item()
print(f'Mean Average Error: {MAEScore}')
MSEScore = nn.functional.mse_loss(preds, labels).item()
print(f'Mean Squared Error: {MSEScore}')

print('done')