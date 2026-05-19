$Suffix = "lab"
$Time = Get-Date -Format "HHmmss"
$JobName = "cs552-benlasso-g54-$Suffix-$Time"

Write-Output ">>> Submitting $JobName  (1 GPU)"

# We simplify the command argument to remove double quotes, dollar signs, and parentheses.
# This prevents Windows CLI argument-parsing mangling.
$CmdString = 'mkdir -p /scratch/hf_cache /scratch/wandb && cd /scratch && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/scratch --ServerApp.token=cs552'

$Arguments = @(
  "submit",
  "--name", $JobName,
  "-p", "course-cs-552-benlasso",
  "--image", "registry.rcp.epfl.ch/course-cs-552/base-vllm:v1",
  "--gpu", "1",
  "--large-shm",
  "--interactive",
  "--node-pools", "a100-40g",
  "--working-dir", "/scratch",
  "--environment", "HF_HOME=/scratch/hf_cache",
  "--environment", "HF_HUB_ENABLE_HF_TRANSFER=1",
  "--environment", "WANDB_DIR=/scratch/wandb",
  "--existing-pvc", "claimname=course-cs-552-scratch-g54,path=/scratch",
  "--existing-pvc", "claimname=course-cs-552-shared-ro,path=/shared-ro",
  "--existing-pvc", "claimname=course-cs-552-shared-rw,path=/shared-rw",
  "--command", "--", "/bin/bash", "-lc", $CmdString
)

& runai @Arguments

Write-Output ">>> Job submitted: $JobName"
Write-Output "Watch it start:    runai describe job $JobName -p course-cs-552-benlasso"
Write-Output "Stream logs:       runai logs -f $JobName -p course-cs-552-benlasso"
Write-Output "When Running:      runai port-forward $JobName --port 8888:8888 -p course-cs-552-benlasso"
