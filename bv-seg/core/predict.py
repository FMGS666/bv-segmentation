import os
import gc
import torch

import pandas as pd 

from patchify import patchify, unpatchify
from collections import defaultdict
from monai.networks.nets import SwinUNETR
from monai.data import Dataset, CacheDataset, ThreadDataLoader, load_decathlon_datalist
from monai.transform import AsDiscrete

from ..src.file_loaders.tif_file_loader import TifFileLoader
from ..src.file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from ..src.data_utils.utils import get_volumes_fold_splits, get_datasets_from_data_path, dump_dataset_metadata
from ..src.feature_engineering.monai_transformations import get_monai_transformations
from ..src.volumes.write_volumes import write_volumes_to_tif


def predict(
        args,
        device
    ) -> None:
    test_data_path = args.test_data_path
    patch_size = args.patch_size
    paths_to_predictors = args.paths_to_predictors
    write_test_volumes = args.write_test_volumes
    feature_size = args.feature_size
    patch_size = args.patch_size
    submission_name = args.submission_name
    submission_path = args.submission_path
    test_volumes_path = args.volumes_path + "_test"
    left_pad = args.left_pad
    right_pad = args.right_pad
    test_metadata_path = os.path.join(
        args.metadata_base_path,
        "test"
    )
    submission_path = os.path.join(
        submission_path,
        f"{submission_name}-submission.csv"
    )
    print(f"{test_metadata_path=}, {test_volumes_path=}")
    try: 
        os.mkdir(test_volumes_path)
        os.mkdir(test_metadata_path)
    except FileExistsError:
        pass

    if write_test_volumes:

        test_datasets_paths = get_datasets_from_data_path(
            test_data_path
        )

        test_iterable_folders = {
            dataset_name: Tif3DVolumeIterableFolder(dataset_path, "test")
            for dataset_name, dataset_path in test_datasets_paths.items()
        }
        _, _, test_transforms = get_monai_transformations(
            patch_size,
            device,
            left_pad=left_pad,
            right_pad=right_pad
        )

        test_groups = {
            dataset_name: {
                0: [
                    {
                        "image": image_path
                    }
                    for _, image_path, _ in iterable_folder
                ]
            } for dataset_name, iterable_folder in test_iterable_folders.items()
        }

        write_volumes_to_tif(
            test_groups,
            None,
            None,
            False,
            subsample = False,
            dump_folder = test_volumes_path
        )
        
        test_volumes = {
            dataset_name: os.path.join(test_volumes_path, dataset_name)
            for dataset_name in test_groups.keys()
        }
        print(f"{test_groups=}")
        print(f"{test_volumes=}")
        for dataset_name, dataset_volumes_path in test_volumes.items():
            splits = os.listdir(dataset_volumes_path)
            splits_paths = [
                os.path.join(dataset_volumes_path, split)
                for split in splits
            ] 
            for split_id, split_path in enumerate(splits_paths):
                volumes = os.listdir(split_path)
                volumes_paths = [
                    os.path.join(split_path, path)
                    for path in volumes
                ]
                print(f"{volumes_paths=}")
                for volume_path in volumes_paths:
                    volumes_files = os.listdir(volume_path)
                    volumes_paths = [
                        {
                            "image": os.path.join(volume_path, volume_file)
                        }
                        for volume_file in volumes_files
                    ]
                    metadata_dir = os.path.join(
                        test_metadata_path,
                        dataset_name
                    )
                    if not os.path.exists(metadata_dir):
                        os.mkdir(metadata_dir)
                    dump_dataset_metadata(
                        metadata_dir,
                        split_id, 
                        volumes_paths,
                        None
                    )
                    
    
    print("loading models")
    weights = [
        torch.load(
            predictor_weights_path
        ) for predictor_weights_path in paths_to_predictors
    ]
    _, _, test_transforms = get_monai_transformations(
        patch_size,
        device
    )
    models_predictions = defaultdict(list)
    dataset_shape = dict()
    for weight in weights:
        model = SwinUNETR(
            img_size=(
                patch_size, 
                patch_size, 
                patch_size
            ),
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
        ).to(device)
        model.load_from(weights = weight)
        model.to(device)
        dataset_names = os.listdir(test_metadata_path)
        predictions = dict()
        for dataset_name in dataset_names:
            dataset_metadata_dir = os.path.join(test_metadata_path, dataset_name)
            dataset_metadata_file = os.listdir(dataset_metadata_dir)[0]
            dataset_metadata_file = os.path.join(
                dataset_metadata_dir,
                dataset_metadata_file
            )
            test_files = load_decathlon_datalist(dataset_metadata_file, True, "training", base_dir = "./")
            test_ds = Dataset(
                data=test_files, transform=test_transforms
            )
            test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1, shuffle=True)
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["image"].cpu()
                    x = x.numpy()
                    dataset_shape[dataset_name] = x.shape
                    x = patchify(x)
                    x = torch.from_numpy(x)
                    x = x.to(device)
                    logit_map = model(x)
                    del x, logit_map
                    gc.collect()
                    torch.cuda.empty_cache()
                    models_predictions[dataset_name].append(logit_map)
    
    post_pred = AsDiscrete(threshold = 0.0)

    def rle_encode(mask):
        pixel = mask.flatten()
        pixel = np.concatenate([[0], pixel, [0]])
        run = np.where(pixel[1:] != pixel[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = ' '.join(str(r) for r in run)
        if rle == '':
            rle = '1 0'
        return rle
        
    submission = []
    for dataset_name, predictions in models_predictions.items():
        target_shape = dataset_shape[dataset_name]
        predictions = [torch.squeeze(prediction) for prediction in predictions]
        predictions = torch.stack(predictions,dim=0)
        predictions = torch.mean(predictions, dim=0)
        predictions = post_pred(predictions)
        predictions = predictions.numpy()
        predictions = unpatchify(predictions,target_shape)
        predictions = torch.from_numpy(predictions)
        predictions = {"label": predictions}
        predictions = test_transforms.inverse(predictions)["label"]
        predictions = predictions.numpy()
        for idx, prediction in enumerate(predictions):
            rle_predictions = rle_encode(prediction)
            id = f"{dataset_name}_000{idx}"
            submission_data = {"id": id, "rle": rle_predictions}
            current_submission = pd.DataFrame(data= submission_data, index=[0])
            submission.append(current_submission)
    submission_df = pd.concat(submission)
    submission_df.to_csv(submission_path, index=False)
    

