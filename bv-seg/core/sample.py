import os

from ..src.file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from ..src.model_selection.k_fold_split import k_fold_split_iterable_folder
from ..src.model_selection.utils import retrieve_filenames_from_split_indexes, retrieve_k_fold_groups
from ..src.data_utils.utils import get_volumes_fold_splits, get_datasets_from_data_path, dump_dataset_metadata
from ..src.volumes.write_volumes import write_volumes_to_tif

def sample(
        args: dict
    ) -> None:
    train_data_path = args.train_data_path
    K = args.K
    random_state = args.random_state
    shuffle = args.shuffle
    splits_metadata_path = args.splits_metadata_path
    volumes_folder = args.volumes_path
    context_length = args.context_length
    n_samples = args.n_samples
    subsample = args.subsample
    # loading the data
    print("retrieving the paths of each individual dataset")
    train_datasets_paths = get_datasets_from_data_path(
        train_data_path
    )
    print("defining iterable folder for training images")
    train_iterable_folders = {
        dataset_name: Tif3DVolumeIterableFolder(dataset_path, "train") 
        for dataset_name, dataset_path in train_datasets_paths.items() if dataset_name != "kidney_3_dense"
    }
    # performing K-Fold split
    print("splitting the training images using K-Fold")
    train_splits_generators = {
        dataset_name: k_fold_split_iterable_folder(
            dataset_iterable_folder,
            K = K,
            random_state = random_state,
            shuffle = shuffle
        ) for dataset_name, dataset_iterable_folder in train_iterable_folders.items()
    }
    train_datasets_splits_paths = {
        dataset_name: retrieve_filenames_from_split_indexes(
            train_iterable_folders[dataset_name],
            dataset_splits
        ) for dataset_name, dataset_splits in train_splits_generators.items()
    }
    # retrieving the individual groups
    train_splits_groups = {
        dataset_name: retrieve_k_fold_groups(dataset_splits)
        for dataset_name, dataset_splits in train_datasets_splits_paths.items()
    }
    print("writing train volumes")
    write_volumes_to_tif(
        train_splits_groups,
        context_length,
        n_samples,
        True,
        subsample = subsample,
        dump_folder = volumes_folder
    )
    print("volumes written")
    # Now we should construct the dataloader from the sampled volumes
    train_volumes = {
        dataset_name: os.path.join(volumes_folder, dataset_name)
        for dataset_name in train_splits_groups.keys()
    }
    train_volumes = {
        dataset_name: get_volumes_fold_splits(train_volumes_directory)
        for dataset_name, train_volumes_directory in train_volumes.items()
    }
    for dataset_name, splits in train_volumes.items():
        for split_id, split_dictionary in splits.items():
            training_paths = split_dictionary["training"]
            validation_paths = split_dictionary["validation"]
            metadata_dir = os.path.join(
                splits_metadata_path,
                dataset_name
            )
            if not os.path.exists(metadata_dir):
                os.mkdir(metadata_dir)
            dump_dataset_metadata(
                metadata_dir,
                split_id,
                training_paths, 
                validation_paths
            )