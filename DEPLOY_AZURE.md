# Deploy TRELLIS.2 to Azure GPU VM for Testing

## Context

We've implemented multi-image conditioning in the TRELLIS.2 pipelines and need a GPU environment to test end-to-end. The app requires CUDA 12.4, 24GB+ VRAM, and several custom CUDA extensions that must be compiled from source.

### Key Implementation Notes

- Pipelines (`trellis2_image_to_3d.py`, `trellis2_texturing.py`) accept `List[Image.Image]`
- Gradio apps use `gr.Gallery` for multi-image upload, which returns `List[tuple[Image.Image, str | None]]` (image + caption tuples)
- App callbacks extract PIL images from tuples before passing to pipeline: `pil_images = [img for img, _ in images]`
- Example scripts wrap single images in a list: `pipeline.run([image])`

---

## Azure VM Configuration

| Setting | Value |
|---------|-------|
| **VM Size** | `Standard_NC24ads_A100_v4` (1x A100 40GB) |
| **OS** | Ubuntu 22.04 LTS Gen2 |
| **OS Disk** | Premium SSD 128 GB |
| **Region** | East US or West US 2 (A100 availability) |
| **Cost** | ~$3.67/hr (use Spot for ~90% savings during testing) |

---

## Deployment Steps

### Step 1: Provision the VM

```bash
# Create resource group
az group create --name trellis-rg --location eastus

# Create VM (Spot instance for cost savings during testing)
az vm create \
  --resource-group trellis-rg \
  --name trellis-vm \
  --size Standard_NC24ads_A100_v4 \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --os-disk-size-gb 128 \
  --priority Spot \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys

# Open port 7860 for Gradio
az network nsg rule create \
  --resource-group trellis-rg \
  --nsg-name trellis-vmNSG \
  --name AllowGradio \
  --priority 1001 \
  --destination-port-ranges 7860 \
  --access Allow \
  --protocol Tcp
```

### Step 2: SSH in and install CUDA 12.4

```bash
ssh azureuser@<VM_PUBLIC_IP>

# Install CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit

# Set paths
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install system deps + Miniconda

```bash
sudo apt-get update && sudo apt-get install -y libjpeg-dev git build-essential

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source ~/.bashrc
```

### Step 4: Clone the repo and run setup

```bash
git clone --recursive https://github.com/HeathSab/TRELLIS.2.git
cd TRELLIS.2

# Full install (~30-60 min, compiles CUDA extensions)
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
conda activate trellis2
```

### Step 5: Set runtime env vars

```bash
echo 'export OPENCV_IO_ENABLE_OPENEXR=1' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Test

```bash
# Test 1: Single-image (backward compat)
python example.py
# Expect: sample.mp4 and sample.glb created without errors

# Test 2: Multi-image (new feature)
python -c "
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()
images = [
    Image.open('assets/example_image/T.png'),
    Image.open('assets/example_image/T.png'),
]
mesh = pipeline.run(images)[0]
print('Multi-image test passed! Mesh vertices:', mesh.vertices.shape)
"

# Test 3: Launch Gradio app
python app.py
# Access at http://<VM_PUBLIC_IP>:7860
# Upload 1 image -> verify works
# Upload 2+ images via Gallery -> verify multi-image conditioning
```

### Step 7: Cleanup (when done testing)

```bash
# Deallocate VM (stops billing for compute, keeps disk)
az vm deallocate --resource-group trellis-rg --name trellis-vm

# Or delete everything
az group delete --name trellis-rg --yes
```

---

## Key Notes

- **First run downloads 16.2 GB** of model weights from HuggingFace — takes a few minutes
- **Spot VMs** can be evicted — fine for testing, not production
- **SSH tunnel alternative** if you don't want to open port 7860: `ssh -L 7860:localhost:7860 azureuser@<IP>`
- If flash-attn fails to install, fallback: `pip install xformers && export ATTN_BACKEND=xformers`
- Total disk usage: ~50 GB (repo + models + conda env + compiled extensions)
