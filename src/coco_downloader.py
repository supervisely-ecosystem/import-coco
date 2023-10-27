import os
import shutil

from os.path import basename, dirname, normpath

import requests
from supervisely.io.fs import download, file_exists, mkdir, silent_remove, dir_exists, get_file_ext
import supervisely as sly

import dl_progress
import globals as g


def download_file_from_link(link, file_name, archive_path, progress_message, app_logger):
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
    download_file_from_link(link, file_name, archive_path, f"Download {file_name}", app_logger)
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
    download_file_from_link(link, file_name, archive_path, f"Download {file_name}", app_logger)
    shutil.unpack_archive(archive_path, save_path, format="zip")
    for file in os.listdir(ann_dir):
        if file != f"instances_{dataset}.json" and file != f"captions_{dataset}.json":
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


def download_dir_from_supervisely(path_to_remote_dir, dir_path, progress_message, app_logger):
    dir_size = g.api.file.get_directory_size(g.TEAM_ID, path_to_remote_dir)
    if not dir_exists(dir_path):
        progress_upload_cb = dl_progress.get_progress_cb(
            g.api, g.TASK_ID, progress_message, total=dir_size, is_size=True
        )
        g.api.file.download_directory(
            g.TEAM_ID, path_to_remote_dir, dir_path, progress_cb=progress_upload_cb
        )

        app_logger.info(f'Directory "{path_to_remote_dir}" has been successfully downloaded')


def find_coco_datasets(path):
    listdir = sly.fs.list_files(path)
    contains_dirs = any(sly.fs.dir_exists(x) for x in listdir)
    contains_imgs = any(sly.image.has_valid_ext(x) for x in listdir)
    return sly.fs.dir_exists(path) and not contains_dirs and contains_imgs


def is_archive(path):
    return get_file_ext(path) in [".zip", ".tar"] or path.endswith(".tar.gz")


def prepare_input_path(api, team_id, input_dir, input_file):
    if input_dir:
        listdir = api.file.listdir(team_id, input_dir)
        archives_cnt = len([is_archive(file) for file in listdir if is_archive(file) is True])
        if archives_cnt > 1:
            raise Exception("Multiple archives are not supported.")
        if len(listdir) == 1 and archives_cnt == 1:
            sly.logger.info(
                "Folder mode is selected, but archive file is uploaded. Switching to file mode."
            )
            input_dir, input_file = None, listdir[0]
        else:
            if any(basename(normpath(x)) in ["images", "annotations"] for x in listdir):
                if dirname(normpath(input_dir)) != "/import/import-coco":
                    input_dir = dirname(normpath(input_dir))
            if basename(normpath(input_dir)) in ["images", "annotations"]:
                if dirname(normpath(input_dir)) != "/import/import-coco":
                    input_dir = dirname(normpath(input_dir))
            if not input_dir.endswith("/"):
                input_dir += "/"

    if input_file:
        file_ext = get_file_ext(input_file)
        if not is_archive(input_file):
            sly.logger.info("File mode is selected, but uploaded file is not archive.")
            if sly.image.is_valid_ext(file_ext) or file_ext == ".json":
                parent_dir = dirname(normpath(input_file))
                listdir = api.file.listdir(team_id, parent_dir)

                if basename(normpath(parent_dir)) in ["images", "annotations"]:
                    parent_dir = dirname(dirname(parent_dir))
                elif ["images", "annotations"] in [basename(normpath(x)) for x in listdir]:
                    parent_dir = dirname(parent_dir)
                if not parent_dir.endswith("/"):
                    parent_dir += "/"
                input_dir, input_file = parent_dir, None
            else:
                raise Exception(
                    "Incorrect data in COCO format. "
                    f"Must contain images and JSON annotations. Got: {input_file}."
                )
    return input_dir, input_file


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
    if remote_path is None or remote_path == "":
        sly.logger.warn(f"Incorrect path to the custom dataset: {remote_path}")
        return []

    g.INPUT_DIR, g.INPUT_FILE = prepare_input_path(g.api, g.TEAM_ID, g.INPUT_DIR, g.INPUT_FILE)

    if g.INPUT_FILE:
        if not g.api.file.exists(g.TEAM_ID, g.INPUT_FILE):
            raise FileNotFoundError(f"File {g.INPUT_FILE} not found in Team Files")
        archive_name = basename(normpath(g.INPUT_FILE))
        archive_path = os.path.join(g.COCO_BASE_DIR, archive_name)
        download_file_from_supervisely(
            g.INPUT_FILE, archive_path, archive_name, f'Download "{archive_name}"', app_logger
        )
        app_logger.info("Unpacking archive...")
        sly.fs.unpack_archive(archive_path, g.COCO_BASE_DIR, remove_junk=True)
        silent_remove(archive_path)
        coco_listdir = os.listdir(g.COCO_BASE_DIR)
        for file in coco_listdir:
            file_path = os.path.join(g.COCO_BASE_DIR, file)
            if is_archive(file_path):
                sly.fs.unpack_archive(file_path, g.COCO_BASE_DIR)
                silent_remove(file_path)

        coco_listdir = os.listdir(g.COCO_BASE_DIR)
        if any(basename(normpath(x)) in ["images", "annotations"] for x in coco_listdir):
            g.COCO_BASE_DIR = dirname(normpath(g.COCO_BASE_DIR))
            sly.logger.info(f"COCO_BASE_DIR: {g.COCO_BASE_DIR}")

    elif g.INPUT_DIR:
        if not g.api.file.dir_exists(g.TEAM_ID, g.INPUT_DIR):
            raise FileNotFoundError(f"Directory {g.INPUT_DIR} not found in Team Files")
        dir_name = basename(normpath(g.INPUT_DIR))
        dir_path = os.path.join(g.COCO_BASE_DIR, dir_name)
        download_dir_from_supervisely(g.INPUT_DIR, dir_path, f'Download "{dir_name}"', app_logger)
    else:
        sly.logger.warn(f"No valid data found in the given path: {remote_path}")
        return []

    sly.fs.remove_junk_from_dir(g.COCO_BASE_DIR)
    coco_datasets = [d for d in sly.fs.dirs_filter(g.COCO_BASE_DIR, find_coco_datasets)]
    return coco_datasets


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
