# volume uniformizing and down-sampling for the STOIC dataset

import os, argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import SimpleITK as sitk
import torch
import torchio as tio


def load_mha(img_path):
    '''
    Load `.MHA` image file from `img_path` and return Simple-ITK image.
    '''
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(img_path)
    image_obj = reader.Execute()
    return image_obj


def preprocess(file: str, out_dir: str):
    '''
    Read .MHA `file` and save processed image tensor to `out_dir`.
    '''
    # define pipeline
    transform = tio.Compose([
        tio.Resample(target=(1.5, 1.5, 1.5), image_interpolation='bspline'),
        tio.Clamp(out_min=-1200, out_max=600),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.CropOrPad(target_shape=(320, 320, 320))
    ])
    # parse patient ID
    _, name = os.path.split(file)
    patient_id = name.split('.')[0]
    # process image
    image = transform(tio.ScalarImage(file)).data
    # save tensor
    out_path = os.path.join(out_dir, patient_id + '.pt')
    torch.save(image, out_path)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    # get data files
    files = [file.path for file in os.scandir(args.in_dir)]
    print('Got %d input files.'%len(files))
    # process images
    Parallel(n_jobs=args.workers)(delayed(preprocess)(file, args.out_dir) for file in tqdm(files))