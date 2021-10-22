import os
import numpy as np
import argparse
import torch
import config

from scipy.spatial.distance import cdist
from glob import glob
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from copy import deepcopy


tfs = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def estimate_trend(cfg, sim_mat, series, splits):
    new_series = []
    keep_k = []
    sim_mat_test = sim_mat[splits == 1, :][:, splits == 0]
    for i in range(sim_mat_test.shape[0]):
        best = np.where(sim_mat_test[i, ...] >= np.sort(sim_mat_test[i, ...])[::-1][cfg.k - 1])[0]
        keep_k.append(deepcopy(best[sim_mat_test[335, best].argsort()[::-1]]))
        new_serie = []
        for s in range(cfg.shuffle):
            np.random.shuffle(best)
            k_n = best[:cfg.k]
            norm_coeff = sim_mat_test[i, k_n].sum()
            new_serie_tmp = np.zeros((series.shape[1],))
            for kk_n in k_n:
                new_serie_tmp = new_serie_tmp + (sim_mat_test[i, kk_n] / norm_coeff) * series[splits == 0, :][kk_n, ...]
            new_serie.append(new_serie_tmp)
        new_serie = np.stack(new_serie).mean(axis=0)
        new_series.append(new_serie)
    new_series = np.stack(new_series)
    return new_series, keep_k


def load_csv():
    import pandas as pd
    train_data = pd.read_csv(config.TRAIN_DATASET, index_col=[0])
    test_data = pd.read_csv(config.TEST_DATASET, index_col=[0])
    norm_scale = int(np.load(config.NORMALIZATION_VALUES_PATH))
    models_dict = {}
    colors_dict = {}
    fabric_dict = {}

    idx_model = 0
    idx_color = 0
    idx_fabric = 0
    tags = []
    series = []
    img_paths = []
    codes = []
    splits = []
    train_codes = train_data.index.values
    for code in train_codes:
        codes.append(code)
        item = train_data.loc[code]
        series.append([item[str(i)] for i in range(12)])
        img_paths.append(item['image_path'])
        model = item['category']
        color = item['color']
        fabric = item['fabric']
        if model not in models_dict:
            models_dict[model] = idx_model
            idx_model += 1
        if color not in colors_dict:
            colors_dict[color] = idx_color
            idx_color += 1
        if fabric not in fabric_dict:
            fabric_dict[fabric] = idx_fabric
            idx_fabric +=1

        tags.append([models_dict[model], colors_dict[color], fabric_dict[fabric]])
        splits.append(0)
    test_codes = test_data.index.values
    for code in test_codes:
        codes.append(code)
        item = test_data.loc[code]
        series.append([item[str(i)] for i in range(12)])
        img_paths.append(item['image_path'])
        model = item['category']
        color = item['color']
        fabric = item['fabric']
        if model not in models_dict:
            models_dict[model] = idx_model
            idx_model += 1
        if color not in colors_dict:
            colors_dict[color] = idx_color
            idx_color += 1
        if fabric not in fabric_dict:
            fabric_dict[fabric] = idx_fabric
            idx_fabric +=1

        tags.append([models_dict[model], colors_dict[color], fabric_dict[fabric]])
        splits.append(1)
    
    tags = np.stack(tags)
    splits = np.stack(splits)
    series = np.stack(series)
    if config.NORM:
        series = series / norm_scale

    return tags, img_paths, codes, series, splits



def eval(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.makedirs(os.path.join(cfg.save_path, cfg.save_tag), exist_ok=True)
    print("-----------------------------------------------------------------------------------------------------------")
    print("Exp modality: {}, Normalize: {}, Window-test: [0, {}]".format(cfg.exp_num, config.NORM, cfg.window_test_end))
    print("Loading dataset...")

    tags, img_paths, codes, series, splits = load_csv()

    similarity_matrix = None

    if cfg.exp_num in [1,3]:
        print("Computing similarity matrix...")
        
        dist1 = np.asarray(cdist(tags[:, 0][:, np.newaxis], tags[:, 0][:, np.newaxis], 'euclidean') != 0, dtype=int)
        dist2 = np.asarray(cdist(tags[:, 1][:, np.newaxis], tags[:, 1][:, np.newaxis], 'euclidean') != 0, dtype=int)
        dist3 = np.asarray(cdist(tags[:, 2][:, np.newaxis], tags[:, 2][:, np.newaxis], 'euclidean') != 0, dtype=int)
        tags_similarity = (dist1 + dist2 + dist3) / 3

        similarity_matrix = tags_similarity

    if cfg.exp_num in [2, 3]:
        if len(glob('features/*/*.npy')) < series.shape[0]:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                incv3 = models.inception_v3(pretrained=True).to(device)
                incv3.fc = Identity()
                incv3.eval()
                features = []
                p_bar = tqdm(desc="Extracting image embedding", total=len(img_paths))
                for im_p in img_paths:
                    if os.path.isfile("features/" + im_p.replace('.png', '.npy')):
                        p_bar.update()
                        continue
                    tmp = Image.open(os.path.join(config.DATASET_PATH, im_p)).convert('RGB')
                    with torch.no_grad():
                        out = incv3(tfs(tmp).unsqueeze(0).to('cuda')).squeeze().detach().cpu().numpy()
                    os.makedirs("features/" + im_p.split('/')[-2], exist_ok=True)
                    np.save("features/" + im_p.replace('.png', '.npy'), out)
                    features.append(out)
                    p_bar.update()
                features = np.stack(features)
                p_bar.close()
        else:
                features = []
                for im_p in img_paths:
                    out = np.load("features/" + im_p.replace('.png', '.npy'))
                    features.append(out)
                features = np.stack(features)
     
        print("Computing similarity matrix...")
        imgs_similarity = cdist(features, features, 'euclidean')
        
        if cfg.exp_num == 3:
            similarity_matrix = 0.5 * imgs_similarity + 0.5 * tags_similarity
        else:
            similarity_matrix = imgs_similarity

    if cfg.window_test_start is None:
        cfg.window_test_start = 0
    if cfg.window_test_end is None:
        cfg.window_test_end = series.shape[1]
    print("Forecasting new series...")
    similarity_matrix = 1 - (
            (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min()))
    new_series, _ = estimate_trend(cfg, similarity_matrix, series, splits)

    pred = new_series[:, cfg.window_test_start:cfg.window_test_end]
    gt = series[splits == 1, cfg.window_test_start:cfg.window_test_end]

    tot_mae = np.mean(np.mean(np.abs(gt - pred), axis=-1))
    tot_wape = 100 * np.sum(np.sum(np.abs(gt - pred), axis=1)) / np.sum(gt)
  
    overestimates = (np.mean(pred - gt, axis=-1) > 0).sum()
    print(f"Overestimates: {overestimates}/{pred.shape[0]}",
          f"Average Overestimates: {np.mean(np.mean(pred[np.mean(pred - gt, axis=-1) > 0, ...] - gt[np.mean(pred - gt, axis=-1) > 0, ...], axis=-1))}",
          f"Std Overestimates: {np.std(np.mean(pred[np.mean(pred - gt, axis=-1) > 0, ...] - gt[np.mean(pred - gt, axis=-1) > 0, ...], axis=-1))}")


    with open(os.path.join(cfg.save_path, cfg.save_tag, 'results.txt'), 'w') as f:
        f.write("Window: [" + str(cfg.window_test_start) + ',' + str(
            cfg.window_test_end) + '] - Tag: ' + cfg.save_tag + '\nMAE: ' + str(
            tot_mae) + '\nwMAPE: ' + str(tot_wape))

    print("Window: [" + str(cfg.window_test_start) + ',' + str(
        cfg.window_test_end) + '] - Tag: ' + cfg.save_tag + '\nMAE: ' + str(
        tot_mae) + '\nWAPE: ' + str(tot_wape))
    print("\n{:.3f}\t{:.3f}".format(
            tot_mae,
            tot_wape).replace('.',','))
    os.makedirs(os.path.join(cfg.save_path, 'results', cfg.save_tag),exist_ok=True)
    torch.save({'results': new_series, 'gts': gt, 'codes': [codes[ii] for ii in np.where(splits == 1)[0]]},
               os.path.join(cfg.save_path, 'res', cfg.save_tag, 'res.pth'))


    print("-----------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KNN Baselines")

    parser.add_argument("--exp_num", type=int, help="1->KNN,2->Embedded KNN with image, 3-> Embedded KNN with all",
                        default=2)

    parser.add_argument('--k', type=int, default=11)
    parser.add_argument('--shuffle', type=int, default=50)
    parser.add_argument('--window_test_start', type=int, default=None)
    parser.add_argument('--window_test_end', type=int, default=12)

    parser.add_argument('--save_path', type=str, default="results")
    parser.add_argument('--save_tag', type=str, default="img_12")

    args = parser.parse_args()

    eval(args)
