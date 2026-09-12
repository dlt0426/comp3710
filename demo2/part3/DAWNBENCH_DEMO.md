# Part 3.2: ResNet-18

The model is implemented from scratch with four stages of two BasicBlocks each
([2, 2, 2, 2]). For 32x32 CIFAR-10 images it uses a 3x3 stride-1 stem without
initial max pooling. This retains the ResNet-18 block structure and depth.

From the `demo2/part3` directory on Rangpur:

```bash
sbatch dawnbench_job.sh --mode train
sbatch dawnbench_job.sh --mode inference
sbatch dawnbench_job.sh --mode demo
```

Wait for training to finish before running inference or demo. The training run
saves `resnet18_cifar10.pt` beside the script. Inference loads it and reports
full-test-set accuracy plus ten example class predictions. Demo first performs
inference, then trains one complete epoch using a fresh SGD optimizer at a small
learning rate. It does not overwrite the trained checkpoint. For a live demo,
run the same Python commands within an allocated GPU session, for example:

```bash
python dawnbench.py --mode demo
```

Training defaults to 40 epochs, batch size 256, SGD with OneCycleLR, label
smoothing, per-image random crop/flip, and CUDA mixed precision. The dataset is
kept on the selected device. Use `--epochs`, `--batch-size`, `--lr`,
`--checkpoint`, `--data-dir`, or `--no-amp` to configure runs.

Accuracy is measured after the final epoch. Printed timing distinguishes
training-only time from training plus final evaluation; both exclude dataset
loading and checkpoint saving. Targets (>90%, 94%, and the 360-second reference)
require actual cluster measurements. No accuracy or runtime is guaranteed.
The checkpoint is for inference/demo, not exact training resumption: optimizer
and scheduler state are not saved.
