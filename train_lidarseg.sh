set -x

# bash launcher.sh config/tpv_lidarseg.py out/tpv_lidarseg

bash launcher.sh config/tpv_fusion_lidarseg.py out/tpv_fusion_lidarseg \
    --resume-from out/tpv_fusion_lidarseg/latest.pth