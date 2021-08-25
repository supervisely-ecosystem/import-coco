import os
import json
import globals as g
import coco_downloader
import supervisely_lib as sly
from supervisely_lib.io.fs import get_file_name_with_ext

from coco_utils import COCOUtils


@g.my_app.callback("import_coco")
@sly.timeit
def import_coco(api: sly.Api, task_id, context, state, app_logger):
    coco_downloader.download_custom_coco_dataset(g.custom_ds, app_logger)
    coco_downloader.download_original_coco_dataset(g.original_ds, app_logger)

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.team_id,
        "WORKSPACE_ID": g.workspace_id
    })
    g.my_app.run(initial_events=[{"command": "import_coco"}])

if __name__ == '__main__':
    sly.main_wrapper("main", main)

