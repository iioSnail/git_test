#python c_train.py --datas wang271k,sighan15train,sighan14train,sighan13train,sighan15train,sighan14train,sighan13train \
#                  --batch-size 32 \
#                  --valid-ratio 0.05 \
#                  --resume \
#                  --epochs 20 \
#                  --model MultiModalMyModel \
#                  --ckpt-dir /root/autodl-tmp/csc/ \
#
#
#python c_train.py --datas wang271k,sighan15train,sighan14train,sighan13train,sighan15train,sighan14train,sighan13train \
#--batch-size 32 \
#--valid-ratio 0.05 \
#--no-resume \
#--epochs 20 \
#--model MultiModalMyModel \
#--eval \
#--workers 8 \
#--test-data sighan2015test \
#--ckpt-dir /root/autodl-tmp/csc_outputs/ \
#--hyper-params bert_base_lr=2e-5

python c_train.py \
  --model scope-p \
  --data medical_entities \
  --data-type word \
  --limit-batches -2 \
  --batch-size 128 \
  --workers 4 \
  --epochs 40 \
  --no-resume \
  --eval \
  --test-data mcsc_sm