import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from image_lib import resize_to_square
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import config
from sklearn.preprocessing import MinMaxScaler

normalization_values = np.load(config.NORMALIZATION_VALUES_PATH)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, use_saved_feat=False):
        self.json_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.use_saved_feat = use_saved_feat


        categ_dict = torch.load(config.CATEG_DICT)
        color_dict = torch.load(config.COLOR_DICT)
        fabric_dict = torch.load(config.FAB_DICT)


        # Category
        self.category = [categ_dict[x] for x in self.json_df['category'].tolist()]
        # Color
        self.color = [color_dict[x] for x in self.json_df['exact_color'].tolist()]
        # Fabric
        self.fabric = [fabric_dict[x] for x in self.json_df['texture'].tolist()]


        # Release date
        self.release_date = self.json_df['release_date'].tolist()

        #Days
        self.days = self.json_df['day'].tolist()
        #Weeks
        self.weeks = self.json_df['week'].tolist()
        #Months
        self.months = self.json_df['month'].tolist()
        #Years
        self.years = self.json_df['year'].tolist()

        # Labels
        new_labels = self.json_df.iloc[:, 1:13].values.tolist()
        self.img_labels = pd.Series(new_labels)
        
        # Path
        self.path = self.json_df["image_path"]
        
        # Transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Image

        img_path = os.path.join(self.img_dir, self.path.iloc[idx])
        image = cv2.imread(img_path)
        image = resize_to_square(image)
        image_2 = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        orig_8x8 = cv2.resize(image_2, (8,8), interpolation = cv2.INTER_AREA)

        # Image feature
        feat_path = os.path.join(config.SAVED_FEATURES_PATH, self.path.iloc[idx].replace(".png", ".pth"))
        img_feature=torch.load(feat_path).squeeze()

        # Category
        category = self.category[idx]
        # Color
        color = self.color[idx]
        
        # Fabric
        fabric = self.fabric[idx]

        # Release date
        release_date = self.release_date[idx]

        #Temporal features
        temporal_features = []
        temporal_features.append(self.days[idx])
        temporal_features.append(self.weeks[idx])
        temporal_features.append(self.months[idx])
        temporal_features.append(self.years[idx])
        temporal_features = torch.as_tensor(temporal_features, dtype=torch.float)


        # Label
        trend = self.img_labels.iloc[idx]
        trend = torch.FloatTensor(trend)
        
        # Applying transform
        if self.transform:
            image_transformed = self.transform(image)
        if self.target_transform:
            trend = self.target_transform(trend)

        return (image_transformed, trend, category, color, fabric, orig_8x8, release_date, temporal_features, img_feature, self.path.iloc[idx])


def resize2d(img, size):
    from torch.autograd import Variable
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

def exog_extractor(date, categ, color, fabric):

    categ = np.asarray(categ)
    color = np.asarray(color)
    fabric = np.asarray(fabric)

    gtrends = pd.read_csv(config.COMPOSED_GTREND, parse_dates=['date'], index_col=[0])

    out_gtrends = []
    weeks = config.EXOG_LEN
    for i in range(categ.shape[0]):
        categ_dict = torch.load(config.CATEG_DICT)
        categ_dict = {v: k for k, v in categ_dict.items()}

        color_dict = torch.load(config.COLOR_DICT)
        color_dict = {v: k for k, v in color_dict.items()}

        fabric_dict = torch.load(config.FAB_DICT)
        fabric_dict = {v: k for k, v in fabric_dict.items()}

        cat = categ_dict[categ[i]]
        col = color_dict[color[i]]
        fab = fabric_dict[fabric[i]]

        start_date = pd.to_datetime(date[i])
        gtrend_start = start_date - pd.DateOffset(weeks=52)
        cat_gtrend = gtrends.loc[gtrend_start:start_date][cat][-52:].values
        col_gtrend = gtrends.loc[gtrend_start:start_date][col][-52:].values
        fab_gtrend = gtrends.loc[gtrend_start:start_date][fab.replace(' ', '')][-52:].values

        cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
        col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
        fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
        
        multitrends = np.hstack([cat_gtrend[:weeks], col_gtrend[:weeks], fab_gtrend[:weeks]]).astype(np.float32)

        out_gtrends.append(multitrends)
    out_gtrends = np.vstack(out_gtrends)
    return out_gtrends