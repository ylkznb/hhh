## Requirements

```bash
# create virtual environment
conda create -n motion python=3.8
conda activate motion
# install packages
pip install -r requirements.txt
```

## Weights of Foundation Models

```bash
git lfs install
## You can choose the ModelScopeT2V or ZeroScope, etc., as the foundation model.
## ZeroScope
git clone [https://huggingface.co/cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w) ./models/zeroscope_v2_576w/
## ModelScopeT2V
git clone [https://huggingface.co/damo-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b) ./models/model_scope/
```

## Motion with Customized Appearance

### Train

Train the spatial path with reference images.

```bash
python train.py --config ./configs/config_multi_images.yaml
```

Then train the temporal path to learn the motions in reference videos.

```bash
python train.py --config ./configs/config_multi_videos.yaml
```

### Inference

Inference with spatial path learned from reference images and temporal path learned form reference videos.

```bash
python inference_multi.py --model /path/to/the/foundation/model --prompt "Your prompt" --spatial_path_folder /path/to/the/trained/Motion/spatial/lora/ --temporal_path_folder /path/to/the/trained/Motion/temporal/lora/ --noise_prior 0
```