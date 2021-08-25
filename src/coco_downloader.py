import os
import shutil
import requests
import dl_progress
import globals as g
from supervisely_lib.io.fs import download, file_exists, silent_remove, mkdir, get_file_name


def download_file_from_link(link, file_name, archive_path, progress_message, app_logger):
    response = requests.head(link, allow_redirects=True)
    sizeb = int(response.headers.get('content-length', 0))
    progress_cb = dl_progress.get_progress_cb(g.api, g.task_id, progress_message, sizeb, is_size=True)
    if not file_exists(archive_path):
        download(link, archive_path, cache=g.my_app.cache, progress=progress_cb)
        dl_progress.reset_progress(g.api, g.task_id)
        app_logger.info(f'{file_name} has been successfully downloaded')


def download_coco_images(dataset, archive_path, save_path, app_logger):
    link = g.images_links[dataset]
    file_name = f"{dataset}.zip"
    download_file_from_link(link, file_name, archive_path, f"Download {file_name}", app_logger)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    os.rename(os.path.join(save_path, dataset), os.path.join(save_path, "images"))
    silent_remove(archive_path)


def download_coco_annotations(dataset, archive_path, save_path, app_logger):
    link = None
    file_name = None
    ann_dir = os.path.join(save_path, "annotations")
    if dataset == "train2014" or dataset == "val2014":
        if os.path.exists(ann_dir):
            return
        link = g.annotations_links["trainval2014"]
        file_name = "trainval2014.zip"

    elif dataset == "train2017" or dataset == "val2017":
        if os.path.exists(ann_dir):
            return
        link = g.annotations_links["trainval2017"]
        file_name = "trainval2017.zip"

    download_file_from_link(link, file_name, archive_path, f"Download {file_name}", app_logger)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    for file in os.listdir(ann_dir):
        if not file == f"instances_{dataset}.json":
           silent_remove(os.path.join(ann_dir, file))
    silent_remove(archive_path)


def download_original_coco_dataset(datasets, app_logger):
    for dataset in datasets:
        dataset_dir = os.path.join(g.coco_base_dir, dataset)
        mkdir(dataset_dir)

        archive_path = dataset_dir + ".zip"
        download_coco_images(dataset, archive_path, dataset_dir, app_logger)
        if not dataset.startswith("test"):
            download_coco_annotations(dataset, archive_path, dataset_dir, app_logger)


def download_file_from_supervisely(path_to_remote_dataset, archive_path, archive_name, progress_message, app_logger):
    file_size = g.api.file.get_info_by_path(g.team_id, path_to_remote_dataset).sizeb
    if not file_exists(archive_path):
        progress_upload_cb = dl_progress.get_progress_cb(g.api,
                                                         g.task_id,
                                                         progress_message,
                                                         total=file_size,
                                                         is_size=True)
        g.api.file.download(g.team_id, path_to_remote_dataset, archive_path, progress_cb=progress_upload_cb)
        app_logger.info(
            f'"{archive_name}" has been successfully downloaded')


def download_custom_coco_dataset(path_to_remote_dataset, app_logger):
    archive_name = os.path.basename(os.path.normpath(path_to_remote_dataset))
    archive_path = os.path.join(g.coco_base_dir, archive_name)
    download_file_from_supervisely(path_to_remote_dataset, archive_path, archive_name, f'Download "{archive_name}"', app_logger)
    shutil.unpack_archive(archive_path, g.coco_base_dir)
    silent_remove(archive_path)
