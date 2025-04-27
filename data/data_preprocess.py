import csv
import ast
import random
import json
from tqdm import tqdm


def parse_topic_tag(s):
    if s is None:
        return ["默认"]
    s = s.strip()
    if s == "[]":
        return ["默认"]
    s = s[1:-1]
    items = [item.strip() for item in s.split(",") if item.strip()]
    if not items:
        return ["默认"]
    return items


user_features = {}
with open("user_features.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = int(row["user_id"])
        feats = []
        for i in range(18):
            val = row.get(f"onehot_feat{i}", "")
            val = val.strip()
            feats.append(int(val) if val.isdigit() else 0)
        user_features[user_id] = feats

item_features = {}
with open("item_categories.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_id = int(row["video_id"])
        feats = ast.literal_eval(row["feat"])
        item_features[video_id] = feats

video_caps_tags = {}
with open("kuairec_caption_category.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_id_str = row["video_id"]
        try:
            video_id = int(video_id_str)
        except (ValueError, TypeError):
            continue
        caption = row["caption"] if row["caption"] else "默认"
        tags = parse_topic_tag(row["topic_tag"])
        video_caps_tags[video_id] = {"caption": caption, "tags": tags}

results = []
with open("big_matrix.csv", encoding="utf-8") as f:
    lines = list(f)
    total = len(lines) - 1
with open("big_matrix.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader, desc="Processing", total=total):
        user_id = int(row["user_id"])
        video_id = int(row["video_id"])
        watch_ratio = float(row["watch_ratio"])
        labels = 1 if watch_ratio >= 2 else 0

        user_feat = user_features.get(user_id, [0] * 18)
        video_feat = item_features.get(video_id, [0])
        cap_tag = video_caps_tags.get(video_id, {"caption": "默认", "tags": ["默认"]})

        result = {
            "user_id": user_id,
            "user_feat": user_feat,
            "video_id": video_id,
            "video_feat": video_feat,
            "caption": cap_tag["caption"],
            "tags": cap_tag["tags"],
            "labels": labels,
        }
        results.append(result)


def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_pos_weight(dataset):
    pos = sum(1 for x in dataset if int(x["labels"]) == 1)
    neg = sum(1 for x in dataset if int(x["labels"]) == 0)
    return (neg if neg > 0 else 1) / (pos if pos > 0 else 1)


user_cnt = len({r["user_id"] for r in results})
video_cnt = len({r["video_id"] for r in results})
max_user_id = max(r["user_id"] for r in results)
max_vid_id = max(r["video_id"] for r in results)
max_user_feat_val = max(max(r["user_feat"]) for r in results)
max_video_feat_val = max(max(r["video_feat"]) for r in results if r["video_feat"])

print(f"不同 user 数量：{user_cnt}")
print(f"不同 video 数量：{video_cnt}")
print(f"user_feat 最大数值：{max_user_feat_val}")
print(f"video_feat 最大数值：{max_video_feat_val}")

random.shuffle(results)

n = len(results)
train_num = int(n * 0.6)
eval_num = int(n * 0.2)

train_dataset = results[:train_num]
eval_dataset = results[train_num : train_num + eval_num]
test_dataset = results[train_num + eval_num :]


save_jsonl("train_dataset.jsonl", train_dataset)
save_jsonl("eval_dataset.jsonl", eval_dataset)
save_jsonl("test_dataset.jsonl", test_dataset)

print(f"Saved train_dataset.jsonl: {len(train_dataset)} 条")
print(f"Saved eval_dataset.jsonl: {len(eval_dataset)} 条")
print(f"Saved test_dataset.jsonl: {len(test_dataset)} 条")


train_pos_weight = get_pos_weight(train_dataset)
eval_pos_weight = get_pos_weight(eval_dataset)
test_pos_weight = get_pos_weight(test_dataset)

print(f"train_dataset 正负样本权重: {train_pos_weight:.4f}")
print(f"eval_dataset 正负样本权重:  {eval_pos_weight:.4f}")
print(f"test_dataset 正负样本权重:  {test_pos_weight:.4f}")

stats = {
    "num_users": user_cnt,
    "num_videos": video_cnt,
    "num_user_features": max_user_feat_val,
    "num_video_features": max_video_feat_val,
    "train_pos_weight": train_pos_weight,
    "eval_pos_weight": eval_pos_weight,
    "test_pos_weight": test_pos_weight,
}

with open("statistics.json", "w", encoding="utf-8") as fp:
    json.dump(stats, fp, ensure_ascii=False, indent=2)
