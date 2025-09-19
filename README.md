# ExecuTorch Demo
WIP image segmentation demo utilizing ExecuTorch and ONNX libraries.

## Development setup:
WSL 2 (Ubuntu) for Python package (ExecuTorch requires Linux or MacOS),

[uv](https://docs.astral.sh/uv/) is used for Python installation and package management,
[ruff](https://docs.astral.sh/ruff/) for linting and formatting, both setup in pyproject.toml. 

Android Studio used on Windows 11, designed for Android API 31 (permissions may not work for 
higher ones currently). An inference model needs to be exported on Python side, and then
pushed to an android device or emulator. adb requires installing Android SDK Platform tools.

```shell
adb shell mkdir /storage/emulated/0/Android/data/com.example.executorchdemo/files/models/
```
```shell
adb push --sync models /storage/emulated/0/Android/data/com.example.executorchdemo/files/
```

