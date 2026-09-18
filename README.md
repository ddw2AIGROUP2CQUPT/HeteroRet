# HeteroRet
HeteroRet: An Image Retrieval Framework with Heterogeneous Training and Inference
* pipeline:
<img width="1731" height="1099" alt="架构图" src="https://github.com/user-attachments/assets/db4cc245-c8c9-4bfb-bfb0-a87d4a6a71b1" />


## Installation

Install Python dependencies:

    pip install -r requirements.txt
    
## Training

Training a delg model:

    python train_delg.py \
        --cfg configs/metric/landmark_vit_8gpu_20251203.yaml \
        OUT_DIR ./output \
        PORT 12001 

## Feature extraction
Extracting global and local feature for multi-scales

    python tools/vit_delg_extract_superglobal.py --cfg configs/landmark_vit_8gpu_20251203.yaml

## Datasets: ROxf and RPar
See(https://github.com/filipradenovic/revisitop) for details

### Results

    cd tools/revisitop
    python my_evaluate_fixed_superglobal.py

  * on roxford5k

**| Backbone | Method | mAP E | mAP M | mAP H |**

  |-----------|-------------|-------|------|-------|
  
  | HeteroRet | Global Only | 87.51 | 69.3 | 42.37 |
  
  | HeteroRet | Global + Re-ranking | **95.18** | **77.31** | **54.6** |

  * on rparis6k(updating)
    All training set version is GLDv2-clean (81313, 1580470)
    
