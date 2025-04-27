
ALBERT_BASE_CHINESE=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/albert-base-chinese-cluecorpussmall
TRAIN_DATASET=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/KuaiRec/KuaiRec/data/train_dataset.jsonl
EVAL_DATASET=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/KuaiRec/KuaiRec/data/eval_dataset.jsonl
TEST_DATASET=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/KuaiRec/KuaiRec/data/test_dataset.jsonl


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py --base_model $ALBERT_BASE_CHINESE --train_dataset $TRAIN_DATASET --eval_dataset $EVAL_DATASET --test_dataset $TEST_DATASET --output_dir /mnt/sdb/zhanglongteng/sdd/zhanglongteng/tmp/kuairec/focal-5e-5 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --num_train_epochs 3 --learning_rate 5e-5
