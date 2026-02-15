# Deploy TRELLIS.2 to Azure GPU VM for Testing

## Context

We've implemented multi-image conditioning in the TRELLIS.2 pipelines and need a GPU environment to test end-to-end. The app requires CUDA 12.4, 24GB+ VRAM, and several custom CUDA extensions that must be compiled from source.

### Key Implementation Notes

- Pipelines (`trellis2_image_to_3d.py`, `trellis2_texturing.py`) accept `List[Image.Image]`
- Gradio apps use `gr.Gallery` for multi-image upload, which returns `List[tuple[Image.Image, str | None]]` (image + caption tuples)
- App callbacks extract PIL images from tuples before passing to pipeline: `pil_images = [img for img, _ in images]`
- Example scripts wrap single images in a list: `pipeline.run([image])`

### Pinned Dependency Versions

| Package | Version | Reason |
|---------|---------|--------|
| `torch` | `2.6.0+cu124` | Pinned in setup.sh, matches CUDA 12.4 |
| `transformers` | `4.56.0` | Must be >=4.56.0 for `DINOv3ViTModel`; versions >=5.0 break RMBG-2.0 (meta tensor error in BiRefNet) |
| `flash-attn` | `2.7.3` | Pinned in setup.sh, compatible with torch 2.6.0 |
| `gradio` | `6.0.1` | Pinned in setup.sh |

---

## Azure VM Configuration

| Setting | Value |
|---------|-------|
| **VM Size** | `Standard_NC24ads_A100_v4` (1x A100 80GB) |
| **OS** | Ubuntu 22.04 LTS Gen2 |
| **OS Disk** | Premium SSD 128 GB |
| **Region** | Australia East (check A100 quota availability in your subscription) |
| **Cost** | ~$3.67/hr standard; Spot ~$0.37-0.73/hr if available |

---

## Deployment Steps

### Step 1: Provision the VM

```bash
# Create resource group
az group create --name trellis-rg --location australiaeast

# Create VM
# Note: Add --priority Spot --eviction-policy Deallocate for cost savings if Spot quota is available
az vm create \
  --resource-group trellis-rg \
  --name trellis-vm \
  --size Standard_NC24ads_A100_v4 \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --os-disk-size-gb 128 \
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

### Step 2: Disable Secure Boot (required for NVIDIA driver)

The default Trusted Launch VM has Secure Boot enabled, which blocks NVIDIA kernel modules from loading. Disable it before installing drivers:

```bash
# Must deallocate first
az vm deallocate --resource-group trellis-rg --name trellis-vm

# Disable Secure Boot
az vm update --resource-group trellis-rg --name trellis-vm \
  --set securityProfile.uefiSettings.secureBootEnabled=false

# Start VM back up
az vm start --resource-group trellis-rg --name trellis-vm
```

### Step 3: SSH in and install NVIDIA driver + CUDA 12.4

```bash
ssh azureuser@<VM_PUBLIC_IP>

# Install NVIDIA driver
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers
sudo reboot
```

Wait ~30 seconds, SSH back in, then verify:

```bash
nvidia-smi
# Should show NVIDIA A100 80GB PCIe
```

Then install CUDA 12.4 toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit

# Set paths
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install system deps + Miniconda

```bash
sudo apt-get install -y libjpeg-dev git build-essential

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source ~/.bashrc
```

### Step 5: Clone the repo and run setup

```bash
git clone --recursive https://github.com/HeathSab/TRELLIS.2.git
cd TRELLIS.2

# Full install (~30-60 min, compiles CUDA extensions)
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
conda activate trellis2
```

### Step 6: Post-setup fixes

```bash
# Set runtime env vars
echo 'export OPENCV_IO_ENABLE_OPENEXR=1' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> ~/.bashrc
source ~/.bashrc

# Fix Pillow/WebP incompatibility (pillow-simd conflicts with newer trimesh GLB export)
pip install --upgrade Pillow

# If flash-attn has ABI issues (undefined symbol errors), use xformers instead:
# pip install xformers
# export ATTN_BACKEND=xformers
```

### Step 7: Test

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

### Step 8: Cleanup (when done testing)

```bash
# Deallocate VM (stops billing for compute, keeps disk)
az vm deallocate --resource-group trellis-rg --name trellis-vm

# Or delete everything
az group delete --name trellis-rg --yes
```

---

## Troubleshooting

### NVIDIA driver won't load after install
Secure Boot blocks unsigned kernel modules. Fix: deallocate VM, disable Secure Boot (Step 2), restart, then reinstall driver.

### `dpkg was interrupted` error
Previous install was interrupted. Fix: `sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a`

### `DINOv3ViTModel` import error
transformers version too old. Requires `>=4.56.0`. Fix: `pip install transformers==4.56.0`

### `Tensor.item() cannot be called on meta tensors` (RMBG-2.0)
transformers version too new (>=5.0). The BiRefNet backbone init uses `torch.linspace().item()` which fails with meta device initialization. Fix: `pip install transformers==4.56.0`

### `flash_attn` undefined symbol / ABI mismatch
flash-attn was compiled against a different torch ABI. Fix: use xformers instead:
```bash
pip install xformers
export ATTN_BACKEND=xformers
```

### `HAVE_WEBPANIM` AttributeError on GLB export
`pillow-simd` (installed by setup.sh) has an outdated WebP module that conflicts with trimesh's GLB exporter. Fix: `pip install --upgrade Pillow` (replaces pillow-simd with standard Pillow).

### Spot VM quota unavailable
Check multiple regions. A100 Spot quota varies. Use standard pricing if Spot isn't available (~$3.67/hr).

---

## Key Notes

- **First run downloads 16.2 GB** of model weights from HuggingFace — takes a few minutes
- **Spot VMs** can be evicted — fine for testing, not production
- **SSH tunnel alternative** if you don't want to open port 7860: `ssh -L 7860:localhost:7860 azureuser@<IP>`
- Total disk usage: ~50 GB (repo + models + conda env + compiled extensions)
- **Deallocate when not testing** — standard VM costs ~$3.67/hr (~$88/day)
