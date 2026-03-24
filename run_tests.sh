#!/bin/bash
# Quick reference for testing multi-view VAE fixes
# Location: /home/piado/projects/aip-lindell/piado/vae/

set -e

echo "=========================================="
echo "Multi-View VAE Fix Testing Script"
echo "=========================================="

PROJECT_ROOT="/home/piado/projects/aip-lindell/piado/vae"
OPENSORA_ROOT="$PROJECT_ROOT/Open-Sora"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Run diagnostic tests
echo -e "\n${YELLOW}Step 1: Running diagnostic tests...${NC}"
cd "$PROJECT_ROOT"
python test_multiview_fixes.py 2>&1 | tee test_results.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Diagnostic tests passed${NC}"
else
    echo -e "${RED}✗ Diagnostic tests failed${NC}"
    exit 1
fi

# Step 2: Check if changes were applied correctly
echo -e "\n${YELLOW}Step 2: Verifying code changes...${NC}"

# Check VAELoss has view_consistency_weight
if grep -q "view_consistency_weight" "$OPENSORA_ROOT/opensora/models/vae/losses.py"; then
    echo -e "${GREEN}✓ VAELoss updated with view_consistency_weight${NC}"
else
    echo -e "${RED}✗ VAELoss missing view_consistency_weight${NC}"
    exit 1
fi

# Check ViewCompressor has learned_weights
if grep -q "use_learned_weights" "$PROJECT_ROOT/DiffSynth-Studio/diffsynth/models/wan_video_vae.py"; then
    echo -e "${GREEN}✓ ViewCompressor updated with learned weights${NC}"
else
    echo -e "${RED}✗ ViewCompressor missing learned weights${NC}"
    exit 1
fi

# Check config has updated settings
if grep -q "view_consistency_weight=0.1" "$OPENSORA_ROOT/configs/vae/train/wan_multiview_finetune.py"; then
    echo -e "${GREEN}✓ Config updated with view_consistency_weight${NC}"
else
    echo -e "${YELLOW}⚠ Config view_consistency_weight not found (may need adjustment)${NC}"
fi

# Step 3: Print summary
echo -e "\n${GREEN}=========================================="
echo "Summary of Changes"
echo "==========================================${NC}"

echo -e "\n${YELLOW}1. Loss Function (opensora/models/vae/losses.py)${NC}"
echo "   - Added view_consistency_weight parameter"
echo "   - Implemented _compute_view_consistency_loss()"
echo "   - Preserves multi-view dimensions in loss computation"

echo -e "\n${YELLOW}2. View Compression (diffsynth/models/wan_video_vae.py)${NC}"
echo "   - ViewCompressor now has learned attention weights"
echo "   - ViewPositionalEmbedding uses multiplicative scaling"

echo -e "\n${YELLOW}3. Configuration (configs/vae/train/wan_multiview_finetune.py)${NC}"
echo "   - view_flatten_in_loss = False"
echo "   - view_consistency_weight = 0.1"

echo -e "\n${GREEN}=========================================="
echo "Next Steps"
echo "==========================================${NC}"

echo "1. Start training:"
echo "   cd $OPENSORA_ROOT"
echo "   python scripts/diffusion/train.py configs/vae/train/wan_multiview_finetune.py"

echo ""
echo "2. Monitor metrics in W&B:"
echo "   - nll_loss (should decrease)"
echo "   - view_consistency_loss (should decrease)"
echo "   - ssim (should reach >0.90)"
echo "   - psnr (should reach >20 dB)"

echo ""
echo "3. Check reconstructed frames for:"
echo "   - Clean per-camera images (no blending)"
echo "   - No ghosting artifacts"
echo "   - Temporal coherence"

echo -e "\n${GREEN}✓ All checks passed. Ready to train!${NC}\n"
