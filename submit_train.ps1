$Time = Get-Date -Format "HHmmss"
$JobName = "cs552-benlasso-g54-train-$Time"

# Edit this for your project. Keep outputs/checkpoints under /scratch.
$TrainCommand = 'if [ ! -d "/scratch/Mock_repo_mnlp" ]; then git clone https://github.com/YoussefBL25/Mock_repo_mnlp.git /scratch/Mock_repo_mnlp; fi && cd /scratch/Mock_repo_mnlp && python3 promptomatix/examples/scripts/multimodal_optimization.py'

Write-Output ">>> Submitting training job $JobName  (1 GPU)"

$Arguments = @(
  "submit",
  "--name", $JobName,
  "-p", "course-cs-552-benlasso",
  "--image", "registry.rcp.epfl.ch/course-cs-552/base-vllm:v1",
  "--gpu", "1",
  "--large-shm",
  "--node-pools", "a100-40g",
  "--working-dir", "/scratch",
  "--environment", "HF_HOME=/scratch/hf_cache",
  "--environment", "HF_HUB_ENABLE_HF_TRANSFER=1",
  "--environment", "WANDB_DIR=/scratch/wandb",
  "--environment", "TRAIN_COMMAND=$TrainCommand",
  "--existing-pvc", "claimname=course-cs-552-scratch-g54,path=/scratch",
  "--existing-pvc", "claimname=course-cs-552-shared-ro,path=/shared-ro",
  "--existing-pvc", "claimname=course-cs-552-shared-rw,path=/shared-rw",
  "--command", "--", "/bin/bash", "-lc", 'mkdir -p /scratch/hf_cache /scratch/wandb /scratch/runs && cd /scratch && eval "$TRAIN_COMMAND"'
)

& runai @Arguments

Write-Output ">>> Training job submitted: $JobName"
Write-Output "Watch it start:    runai describe job $JobName -p course-cs-552-benlasso"
Write-Output "Stream logs:       runai logs -f $JobName -p course-cs-552-benlasso"
