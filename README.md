<div align="center" markdown>
<img src="https://i.imgur.com/KIRxlH0.png"/>


# Import COCO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/import-coco)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-coco)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-coco&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

App converts selected [COCO format](https://cocodataset.org/#home) datasets to [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi) project

Application key points:  
- Supports only **Object Detection** from **COCO** format
- Backward compatible with [Export to COCO](https://github.com/supervisely-ecosystem/export-to-coco)


# How to Use
1. Add [Import COCO](https://ecosystem.supervise.ly/apps/import-coco) to your team from Ecosystem.

[comment]: <> (<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-coco" src="https://imgur.com/QQTBz0C.png" width="350px" style='padding-bottom: 20px'/>  )

2. Run app from the `Plugins & Apps` chapter:

[comment]: <> (<img src="https://imgur.com/6nPIM21.png"/>)

3. Select import mode:

- Your can download selected datasets from [COCO](https://cocodataset.org/#download).  
- Use your custom dataset in COCO format by path to your archive in `Team Files`.

[comment]: <> (<img src="https://imgur.com/8lzZUPc.png" width="600px"/>)

4. After pressing the `Run` button you will be redirected to the `Tasks` page.

# Results

Result project will be saved in your current `Workspace` 

[comment]: <> (with name `mot_video`.)

[comment]: <> (<img src="https://i.imgur.com/tA0lrEN.png"/>)
