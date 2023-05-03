python c_train.py --datas wang271k,sighan15train,sighan14train,sighan13train,sighan15train,sighan14train,sighan13train \
                  --batch-size 32 \
                  --valid-ratio 0.05 \
                  --resume \
                  --epochs 20 \
                  --model MultiModalMyModel \
                  --ckpt-dir /root/autodl-tmp/csc/ \
