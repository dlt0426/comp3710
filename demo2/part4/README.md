ssh s4906926@rangpur.compute.eait.uq.edu.au

cd ~/comp3710

git pull --ff-only origin main

Part 3.2：inference + one training epoch

cd demo2/part3

sbatch dawnbench_job.sh --mode demo

Part 4 Task 2：UNet inference

cd ../part4

sbatch task2_unet_job.sh --mode inference --checkpoint unet_oasis.pt

output:

cat ~/comp3710/demo2/part3/dawnbench_JOBID.out

cat ~/comp3710/demo2/part4/unet_JOBID.out


