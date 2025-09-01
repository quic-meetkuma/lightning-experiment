### Command to test:

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For single device
python train_single_device.py

# For DDP device
QAIC_VISIBLE_DEVICES=0,1 python train_ddp.py