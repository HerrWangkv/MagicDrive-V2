#### Install `SegFormer` (Skip if already installed)

:warning: SegFormer relies on `mmcv-full=1.2.7`, which relies on `pytorch=1.8` (pytorch<1.9). Hence, a separate docker is required.

```shell
docker build -f Dockerfile.segformer -t segformer:latest .
```

Download the pretrained model `segformer.b5.1024x1024.city.160k.pth` from the google_drive / one_drive links in https://github.com/NVlabs/SegFormer#evaluation .

Remember the location where you download into, and pass it to the script in the next step with `--checkpoint`.

<details>
<summary>Troubleshooting: SegFormer Checkpoint Download</summary>

If you encounter problems downloading the original SegFormer checkpoint from the official links, you can alternatively download a backup copy using command: `gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj`
</details>

#### Run Mask Extraction Script

```shell
bash prepare_segformer.sh
bash extract_masks.sh
```
