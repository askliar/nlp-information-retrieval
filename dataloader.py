import pandas as pd
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from imgloader import visual_feat_mapping
from imgloader import img_features

class ImageTextDataSet(Dataset):

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.imagetext_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.imagetext_frame)

    def __getitem__(self, idx):
        text = self.imagetext_frame.iloc[idx, 0].split(' \n ')
        img_id = self.imagetext_frame.iloc[idx, 1]
        target = self.imagetext_frame.iloc[idx, 2]
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        sample = {'text': text, 'img_features': img_feat, 'target': target}
        return sample



data_train = ImageTextDataSet(csv_file='./data/train_data_easy_bow.csv')
dataloader_train = DataLoader(data_train, batch_size=4,
                        shuffle=True, num_workers=4)

data_val = ImageTextDataSet(csv_file='./data/val_data_easy_bow.csv')
dataloader_val = DataLoader(data_val, num_workers=4)

data_test = ImageTextDataSet(csv_file='./data/test_data_easy_bow.csv')
dataloader_test = DataLoader(data_test, num_workers=4)


#

#
