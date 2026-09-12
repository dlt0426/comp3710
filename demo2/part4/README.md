Part 4 Task 2：UNet inference

cd demo2/part4

sbatch task2_unet_job.sh --mode inference --checkpoint unet_oasis.pt

Part 3.2：inference + one training epoch

cd ../part3

sbatch dawnbench_job.sh --mode demo

output:

tail -f unet_JOBID.out

tail -f dawnbench_JOBID.out
