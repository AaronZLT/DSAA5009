import csv
import ast
import random
import json
from tqdm import tqdm


def parse_topic_tag(s: str):
    if s is None:
        return ["默认"]
    s = s.strip()
    if s == "[]":
        return ["默认"]
    s = s[1:-1]
    items = [item.strip() for item in s.split(",") if item.strip()]
    return items or ["默认"]


def save_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_pos_weight(dataset):
    pos = sum(1 for x in dataset if int(x["labels"]) == 1)
    neg = sum(1 for x in dataset if int(x["labels"]) == 0)
    return (neg if neg > 0 else 1) / (pos if pos > 0 else 1)


user_features = {}
with open("user_features.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = int(row["user_id"])
        feats = []
        for i in range(18):
            v = row.get(f"onehot_feat{i}", "").strip()
            feats.append(int(v) if v.isdigit() else 0)
        user_features[uid] = feats

item_features = {}
with open("item_categories.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_features[int(row["video_id"])] = ast.literal_eval(row["feat"])

video_caps_tags = {}
with open("kuairec_caption_category.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            vid = int(row["video_id"])
        except (TypeError, ValueError):
            continue
        caption = row["caption"] if row["caption"] else "默认"
        tags = parse_topic_tag(row["topic_tag"])
        video_caps_tags[vid] = {"caption": caption, "tags": tags}

daily_features = {}
with open("item_daily_features.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = int(row["video_id"])
        date_str = row["date"].strip()
        daily_features[(vid, date_str)] = {
            "date": date_str,
            "play_progress": float(row["play_progress"])
            if row["play_progress"]
            else 0.0,
            "like_cnt": int(row["like_cnt"]) if row["like_cnt"] else 0,
            "comment_cnt": int(row["comment_cnt"]) if row["comment_cnt"] else 0,
            "follow_cnt": int(row["follow_cnt"]) if row["follow_cnt"] else 0,
        }

results = []
missing_pairs = []

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

        raw_date = row["date"].strip()  # e.g. 20200705.0
        date_str = str(int(float(raw_date)))  # → "20200705"

        daily_key = (video_id, date_str)
        daily_feat = daily_features.get(daily_key)
        if daily_feat is None:
            missing_pairs.append(daily_key)
            daily_feat = {
                "date": date_str,
                "play_progress": 0.0,
                "like_cnt": 0,
                "comment_cnt": 0,
                "follow_cnt": 0,
            }

        result = {
            "user_id": user_id,
            "user_feat": user_feat,
            "video_id": video_id,
            "video_feat": video_feat,
            "caption": cap_tag["caption"],
            "tags": cap_tag["tags"],
            "labels": labels,
            "date": daily_feat["date"],
            "play_progress": daily_feat["play_progress"],
            "like_cnt": daily_feat["like_cnt"],
            "comment_cnt": daily_feat["comment_cnt"],
            "follow_cnt": daily_feat["follow_cnt"],
        }
        results.append(result)

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
print(f"Saved eval_dataset.jsonl : {len(eval_dataset)} 条")
print(f"Saved test_dataset.jsonl : {len(test_dataset)} 条")

print(
    f"\n在 item_daily_features 中未找到的 (video_id, date) 共有 {len(missing_pairs)} 条示例："
)
for vid, d in missing_pairs[:20]:
    print(f"video_id={vid}, date={d}")

train_pos_weight = get_pos_weight(train_dataset)
eval_pos_weight = get_pos_weight(eval_dataset)
test_pos_weight = get_pos_weight(test_dataset)

stats = {
    "num_users": len({r["user_id"] for r in results}),
    "num_videos": len({r["video_id"] for r in results}),
    "num_user_feats": max(max(r["user_feat"]) for r in results),
    "num_video_feats": max(max(r["video_feat"]) for r in results if r["video_feat"]),
    "train_pos_weight": train_pos_weight,
    "eval_pos_weight": eval_pos_weight,
    "test_pos_weight": test_pos_weight,
}

with open("statistics.json", "w", encoding="utf-8") as fp:
    json.dump(stats, fp, ensure_ascii=False, indent=2)

print("\nstatistics of the data have been dumped to statistics.json")
