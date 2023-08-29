import os
import shutil

from os.path import basename, dirname, normpath

import requests
from supervisely.io.fs import download, file_exists, mkdir, silent_remove, dir_exists
import supervisely as sly

import dl_progress
import globals as g


def download_file_from_link(
    link, file_name, archive_path, progress_message, app_logger
):
    response = requests.head(link, allow_redirects=True)
    sizeb = int(response.headers.get("content-length", 0))
    progress_cb = dl_progress.get_progress_cb(
        g.api, g.TASK_ID, progress_message, sizeb, is_size=True
    )
    if not file_exists(archive_path):
        download(link, archive_path, cache=g.my_app.cache, progress=progress_cb)
        dl_progress.reset_progress(g.api, g.TASK_ID)
        app_logger.info(f"{file_name} has been successfully downloaded")


def download_coco_images(dataset, archive_path, save_path, app_logger):
    link = g.images_links[dataset]
    file_name = f"{dataset}.zip"
    download_file_from_link(
        link, file_name, archive_path, f"Download {file_name}", app_logger
    )
    shutil.unpack_archive(archive_path, save_path, format="zip")
    os.rename(os.path.join(save_path, dataset), os.path.join(save_path, "images"))
    silent_remove(archive_path)


def download_coco_annotations(dataset, archive_path, save_path, app_logger):
    link = None
    file_name = None
    ann_dir = os.path.join(save_path, "annotations")
    if dataset in ["train2014", "val2014"]:
        if os.path.exists(ann_dir):
            return
        link = g.annotations_links["trainval2014"]
        file_name = "trainval2014.zip"
    elif dataset in ["train2017", "val2017"]:
        if os.path.exists(ann_dir):
            return
        link = g.annotations_links["trainval2017"]
        file_name = "trainval2017.zip"
    download_file_from_link(
        link, file_name, archive_path, f"Download {file_name}", app_logger
    )
    shutil.unpack_archive(archive_path, save_path, format="zip")
    for file in os.listdir(ann_dir):
        if file != f"instances_{dataset}.json":
            silent_remove(os.path.join(ann_dir, file))
    silent_remove(archive_path)


def download_original_coco_dataset(datasets, app_logger):
    for dataset in datasets:
        dataset_dir = os.path.join(g.COCO_BASE_DIR, dataset)
        mkdir(dataset_dir)
        archive_path = f"{dataset_dir}.zip"
        download_coco_images(dataset, archive_path, dataset_dir, app_logger)
        if not dataset.startswith("test"):
            download_coco_annotations(dataset, archive_path, dataset_dir, app_logger)
    return datasets


def download_dir_from_supervisely(
    path_to_remote_dir, dir_path, progress_message, app_logger
):
    dir_size = g.api.file.get_directory_size(g.TEAM_ID, path_to_remote_dir)
    if not dir_exists(dir_path):
        progress_upload_cb = dl_progress.get_progress_cb(
            g.api, g.TASK_ID, progress_message, total=dir_size, is_size=True
        )
        g.api.file.download_directory(
            g.TEAM_ID, 
            path_to_remote_dir, 
            dir_path, 
            progress_cb=progress_upload_cb
        )

        app_logger.info(f'Directory "{path_to_remote_dir}" has been successfully downloaded')


def download_file_from_supervisely(
    remote_path, archive_path, archive_name, progress_message, app_logger
):
    file_size = g.api.file.get_info_by_path(g.TEAM_ID, remote_path).sizeb
    if not file_exists(archive_path):
        progress_upload_cb = dl_progress.get_progress_cb(
            g.api, g.TASK_ID, progress_message, total=file_size, is_size=True
        )
        g.api.file.download(
            g.TEAM_ID,
            remote_path,
            archive_path,
            progress_cb=progress_upload_cb,
        )
        app_logger.info(f'"{archive_name}" has been successfully downloaded')


def download_custom_coco_dataset(remote_path: str, app_logger):
    available_archive_formats = list(zip(*shutil.get_archive_formats()))[0]
    file_ext = sly.fs.get_file_ext(remote_path)
    if g.INPUT_FILE:
        if file_ext.lstrip(".") in available_archive_formats:
            app_logger.info(f"INPUT_FILE is an archive: {remote_path}")
        elif file_ext in sly.image.SUPPORTED_IMG_EXTS + [".json"]:
            app_logger.info(f"INPUT_FILE is not an archive: {remote_path}")
            dir_path = dirname(normpath(remote_path))
            if basename(normpath(dir_path)) in ["images", "annotations"]:
                dir_path = dirname(dirname(dir_path))
            listdir = g.api.file.listdir(g.TEAM_ID, dir_path)
            if ["images", "annotations"] in [basename(normpath(x)) for x in listdir]:
                dir_path = dirname(dir_path)
            app_logger.info(f"Switching to folder mode. INPUT_DIR: {dir_path}")
            g.INPUT_FILE, g.INPUT_DIR = None, dir_path
            app_logger.info(f"Switching to folder mode. INPUT_DIR: {dir_path}")
        else:
            raise ValueError(
                "Incorrect project structure. "
                f"File mode is chosen, but file {remote_path} is not an archive. "
                "Please, read apps overview and prepare the dataset correctly."
            )
    elif g.INPUT_DIR:
        listdir = g.api.file.listdir(g.TEAM_ID, remote_path)
        if sly.fs.get_file_ext(listdir[0]).lstrip(".") in available_archive_formats:
            g.INPUT_DIR, g.INPUT_FILE = None, os.path.join(remote_path, listdir[0])
            app_logger.info(f"Switching to file mode. INPUT_FILE: {g.INPUT_FILE}")
        elif any(basename(normpath(x)) in ["images", "annotations"] for x in listdir):
            g.INPUT_DIR = dirname(normpath(remote_path))
            app_logger.info(f"INPUT_DIR: {g.INPUT_DIR}")
        elif basename(normpath(remote_path)) in ["images", "annotations"]:
            g.INPUT_DIR = dirname(dirname(normpath(remote_path)))
            app_logger.info(f"INPUT_DIR: {g.INPUT_DIR}")

    if g.INPUT_FILE:
        if not g.api.file.exists(g.TEAM_ID, g.INPUT_FILE):
            raise FileNotFoundError(f"File {g.INPUT_FILE} not found in Team Files")
        archive_name = basename(normpath(g.INPUT_FILE))
        archive_path = os.path.join(g.COCO_BASE_DIR, archive_name)
        download_file_from_supervisely(
            g.INPUT_FILE, archive_path, archive_name, f'Download "{archive_name}"', app_logger
        )
        app_logger.info("Unpacking archive...")
        sly.fs.unpack_archive(archive_path, g.COCO_BASE_DIR)
        silent_remove(archive_path)
        assert len(os.listdir(g.COCO_BASE_DIR)) == 1, \
            "ERROR: Archive must contain only 1 project folder with datasets in COCO format."
        app_logger.info("Archive has been unpacked.")
        g.COCO_BASE_DIR = os.path.join(g.COCO_BASE_DIR, os.listdir(g.COCO_BASE_DIR)[0])
    elif g.INPUT_DIR:
        if not g.api.file.dir_exists(g.TEAM_ID, g.INPUT_DIR):
            raise FileNotFoundError(f"Directory {g.INPUT_DIR} not found in Team Files")
        dir_name = basename(normpath(g.INPUT_DIR))
        dir_path = os.path.join(g.COCO_BASE_DIR, dir_name)
        download_dir_from_supervisely(g.INPUT_DIR, dir_path, f'Download "{dir_name}"', app_logger)
        g.COCO_BASE_DIR = os.path.join(g.COCO_BASE_DIR, dir_name)
        sly.fs.remove_junk_from_dir(g.COCO_BASE_DIR)
    else:
        raise ValueError(f"No valid projects found in the given path: {remote_path}")
    return list(os.listdir(g.COCO_BASE_DIR))


def start(app_logger):
    project_name = g.OUTPUT_PROJECT_NAME
    if g.is_original:
        coco_datasets = download_original_coco_dataset(g.original_ds, app_logger)
        if project_name is None or project_name == "":
            project_name = "Original COCO"
    else:
        coco_datasets = download_custom_coco_dataset(g.custom_ds, app_logger)
        if project_name is None or project_name == "":
            project_name = "Custom COCO"
    return project_name, coco_datasets
