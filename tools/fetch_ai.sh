#!/bin/sh
# Downloads the AI models into third_party/models (ncnn file format; they run on
# the player's own Vulkan runner, no ncnn library needed).
#   tools/fetch_ai.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party/dl third_party/models/realesrgan third_party/models/rife
cd third_party/dl

# Real-ESRGAN models (BSD-3-Clause), ncnn format.
[ -f esrgan.zip ] || curl -L -o esrgan.zip https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
rm -rf x_esr && unzip -q esrgan.zip -d x_esr && cp x_esr/models/* ../models/realesrgan/

# RIFE v4 models (MIT), ncnn format, from TNTwise/rife-ncnn-vulkan.
[ -f rife.zip ] || curl -L -o rife.zip https://github.com/TNTwise/rife-ncnn-vulkan/releases/download/20250112/windows.zip
rm -rf x_rife && unzip -q rife.zip -d x_rife
R=$(dirname "$(find x_rife -name rife-ncnn-vulkan.exe | head -1)")
for m in rife-v4.26 rife-v4.25-lite; do rm -rf "../models/rife/$m"; cp -r "$R/$m" ../models/rife/; done
cp "$R/LICENSE" ../models/rife/LICENSE-rife-ncnn-vulkan
echo "done: third_party/models (Real-ESRGAN, RIFE)"
cd ../..

# RobustVideoMatting (GPL-3.0) from the official ONNX, converted with pnnx. The sizes
# baked in at 1080p are rewritten per video by crates/brp-ai/src/matting.rs.
# (Needs cargo for the small ONNX patcher.)
cd third_party/dl
mkdir -p ../models/rvm x_rvm
[ -f rvm_fp32.onnx ] || curl -L -o rvm_fp32.onnx https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
[ -f pnnx.zip ] || curl -L -o pnnx.zip https://github.com/pnnx/pnnx/releases/download/20261020/pnnx-20261020-windows.zip
rm -rf x_pnnx && unzip -q pnnx.zip -d x_pnnx
PNNX=$(find "$PWD/x_pnnx" -name 'pnnx*' -type f | grep -E 'pnnx(\.exe)?$' | head -1)
(cd ../.. && cargo run -q --release -p brp-ai --example onnxfix -- third_party/dl/rvm_fp32.onnx third_party/dl/x_rvm/rvm_1080.onnx \
  "1,3,1080,1920;1,16,135,240;1,20,68,120;1,40,34,60;1,64,17,30" 0.25)
cd x_rvm && "$PNNX" rvm_1080.onnx fp16=0 > pnnx.log 2>&1
cp rvm_1080.ncnn.param ../../models/rvm/rvm_mobilenetv3.param
cp rvm_1080.ncnn.bin ../../models/rvm/rvm_mobilenetv3.bin
echo "done: third_party/models/rvm"

# Real-ESRGAN x2plus ("Best 2x", BSD-3-Clause): official PyTorch weights written
# into the x4plus ncnn graph (same network, 2x2-folded input).
cd ..  # back to third_party/dl
[ -f RealESRGAN_x2plus.pth ] || curl -L -o RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
rm -rf x_x2 && mkdir x_x2 && unzip -q RealESRGAN_x2plus.pth -d x_x2
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- third_party/dl/x_x2 \
  third_party/models/realesrgan/realesrgan-x4plus.param \
  third_party/models/realesrgan/realesrgan-x2plus.param third_party/models/realesrgan/realesrgan-x2plus.bin)
echo "done: realesrgan-x2plus"

# Real-ESRGAN general-x4v3 ("Natural 4x", BSD-3-Clause): compact network trained on
# real-world pictures, and its twin trained without denoising; the player mixes their
# weights for the denoise strength (Real-ESRGAN's -dn). Graph built by pth2ncnn.
cd third_party/dl
for m in realesr-general-x4v3 realesr-general-wdn-x4v3; do
  [ -f $m.pth ] || curl -L -o $m.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/$m.pth
  rm -rf x_$m && mkdir x_$m && unzip -q $m.pth -d x_$m
  (cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- compact third_party/dl/x_$m \
    third_party/models/realesrgan/$m.param third_party/models/realesrgan/$m.bin)
done
rm -f ../models/realesrgan/realesr-general-wdn-x4v3.param
cd ../..
echo "done: realesr-general-x4v3 (+ wdn)"

# 2xNomosUni_compact_otf_medium ("Natural 2x", Helaman / Philip Hofmann, CC-BY-4.0,
# OpenModelDB): compact 2x for real-world pictures. Its weights are stored
# channels-last; pth2ncnn reads them through their strides.
cd third_party/dl
f=2xNomosUni_compact_otf_medium.pth
[ -f $f ] || curl -L -o $f "https://drive.usercontent.google.com/download?id=1lUDklqGCq3fiGO0EBN8P5W-bSsldX1st&export=download&confirm=t"
echo "d9f767194d5163e5b1604ccd87f24f12f23562ed66245bba058eea1f4686ca5a  $f" | sha256sum -c
rm -rf x_nomosuni && mkdir x_nomosuni && unzip -q $f -d x_nomosuni
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- compact third_party/dl/x_nomosuni \
  third_party/models/realesrgan/2xNomosUni_compact_otf_medium.param third_party/models/realesrgan/2xNomosUni_compact_otf_medium.bin)
cd ../..
echo "done: 2xNomosUni_compact_otf_medium"

# SwinIR-M real-world x4 GAN ("Transformer 4x", JingyunLiang/SwinIR, Apache-2.0):
# converted to our runner's layers (ChannelNorm, WindowAttention, GELU).
cd third_party/dl
f=003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
[ -f $f ] || curl -L -o $f "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/$f"
echo "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6  $f" | sha256sum -c
rm -rf x_swinir && mkdir x_swinir && unzip -q $f -d x_swinir
mkdir -p ../models/swinir
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- swinir third_party/dl/x_swinir \
  third_party/models/swinir/SwinIR-M_x4_GAN.param third_party/models/swinir/SwinIR-M_x4_GAN.bin)
cd ../..
echo "done: SwinIR-M_x4_GAN"

# 4xNomos2_hq_dat2 ("Transformer HQ 4x", Philip Hofmann, CC-BY-4.0; DAT architecture,
# zhengchen1999/DAT, Apache-2.0): converted from the author's ONNX export, which kept
# the PyTorch weight names (BatchNorms folded, position bias tables precomputed).
cd third_party/dl
f=4xNomos2_hq_dat2_fp32.onnx
[ -f $f ] || curl -L -o $f "https://github.com/Phhofm/models/releases/download/4xNomos2_hq_dat2/$f"
echo "f5b0b05f767ebc4b85431df6743f52995824eb3f965a8b9b97d94e5b896df5ca  $f" | sha256sum -c
mkdir -p ../models/dat
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- dat third_party/dl/$f \
  third_party/models/dat/4xNomos2_hq_dat2.param third_party/models/dat/4xNomos2_hq_dat2.bin)
cd ../..
echo "done: 4xNomos2_hq_dat2"

# 4xRealWebPhoto_v4_dat2 ("Transformer Restore 4x", Philip Hofmann, CC-BY-4.0): the same
# DAT network trained on degraded pictures. Its ONNX export renamed the weights, so it
# is converted from the .safetensors; the input mean (DIV2K) is given.
cd third_party/dl
f=4xRealWebPhoto_v4_dat2.safetensors
[ -f $f ] || curl -L -o $f "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_dat2/$f"
echo "819047acf43073f63ea7289ed94d6991ba8446a3bdf38e274fadfcbaa2b7381d  $f" | sha256sum -c
mkdir -p ../models/dat
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- dat third_party/dl/$f \
  third_party/models/dat/4xRealWebPhoto_v4_dat2.param third_party/models/dat/4xRealWebPhoto_v4_dat2.bin 0.4488,0.4371,0.404)
cd ../..
echo "done: 4xRealWebPhoto_v4_dat2"

# SEA-RAFT M, Spring fine-tune (princeton-vl/SEA-RAFT, BSD-3-Clause): the "SEA-RAFT (AI)"
# motion option. Three graphs for brp_ai::flow (features, context, one refinement step).
cd third_party/dl
f=searaft_Tartan-C-T-TSKH-spring540x960-M.safetensors
[ -f $f ] || curl -L -o $f "https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/resolve/main/model.safetensors"
echo "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102  $f" | sha256sum -c
mkdir -p ../models/flow
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- searaft third_party/dl/$f third_party/models/flow/searaft_m)
cd ../..
echo "done: SEA-RAFT"

# WAFT a1, TartanAir-Chairs-Things fine-tune (princeton-vl/WAFT, BSD-3-Clause): the
# "WAFT (AI)" motion option. A PyTorch checkpoint (a zip), converted from its pieces into
# six graphs (DINOv2 patches, blocks and DPT head; features; hidden state; refinement)
# plus the position embeddings.
cd third_party/dl
f=waft_a1.pth
[ -f $f ] || curl -L -o $f "https://drive.usercontent.google.com/download?id=1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC&export=download&confirm=t"
echo "9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f  $f" | sha256sum -c
rm -rf x_waft && unzip -q $f -d x_waft
mkdir -p ../models/flow
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- waft third_party/dl/x_waft third_party/models/flow/waft_a1)
cd ../..
echo "done: WAFT"

# NAFNet REDS width 64 ("Deblur", megvii-research/NAFNet, MIT): video frames with
# motion blur and compression. Graph by pth2ncnn (channel attention as the official
# test setting: a local box average, BoxMean).
cd third_party/dl
f=NAFNet-REDS-width64.pth
[ -f $f ] || curl -L -o $f "https://drive.usercontent.google.com/download?id=14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X&export=download&confirm=t"
echo "175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3  $f" | sha256sum -c
rm -rf x_nafnet && mkdir x_nafnet && unzip -q $f -d x_nafnet
mkdir -p ../models/nafnet
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- nafnet third_party/dl/x_nafnet third_party/models/nafnet/nafnet_reds_w64)
cd ../..
echo "done: NAFNet"

# SCUNet real-world PSNR model ("Compression Clean-up", cszn/SCUNet, Apache-2.0).
cd third_party/dl
f=scunet_color_real_psnr.pth
[ -f $f ] || curl -L -o $f "https://github.com/cszn/KAIR/releases/download/v1.0/$f"
echo "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e  $f" | sha256sum -c
rm -rf x_scunet && mkdir x_scunet && unzip -q $f -d x_scunet
mkdir -p ../models/scunet
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- scunet third_party/dl/x_scunet third_party/models/scunet/scunet_color_real_psnr)
cd ../..
echo "done: SCUNet"

# Video Depth Anything Small and Depth Anything V2 Small ("Depth", Apache-2.0; their
# Base and Large models are CC-BY-NC-4.0 and are not used). Graphs by pth2ncnn depth.
cd third_party/dl
mkdir -p ../models/depth
for m in "video_depth_anything_vits 13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609 Video-Depth-Anything-Small vda_vits" \
         "depth_anything_v2_vits 715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378 Depth-Anything-V2-Small da2_vits"; do
  set -- $m
  [ -f $1.pth ] || curl -L -o $1.pth "https://huggingface.co/depth-anything/$3/resolve/main/$1.pth"
  echo "$2  $1.pth" | sha256sum -c
  rm -rf x_$1 && mkdir x_$1 && unzip -q $1.pth -d x_$1
  (cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- depth third_party/dl/x_$1 third_party/models/depth/$4)
done
cd ../..
echo "done: depth"

# BiRefNet lite matting ("Matting", model BiRefNet; ZhengPeng7/BiRefNet, MIT), as one
# graph at 1024 x 1024 by pth2ncnn birefnet.
cd third_party/dl
f=birefnet_lite_matting.safetensors
[ -f $f ] || curl -L -o $f "https://huggingface.co/ZhengPeng7/BiRefNet_lite-matting/resolve/99c33412e3f58e1f33187abdc8c435c645243690/model.safetensors"
echo "ce8bcfc045e336322c0424a5863dcfb7e9ce8fed0a5fd4d1b2b20adf12d97243  $f" | sha256sum -c
mkdir -p ../models/birefnet
(cd ../.. && cargo run -q --release -p brp-ai --example pth2ncnn -- birefnet third_party/dl/$f third_party/models/birefnet/birefnet_lite_matting)
cd ../..
echo "done: BiRefNet"
