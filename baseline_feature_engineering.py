#!/usr/bin/env python
# coding: utf-8

# @Copyright IQIYI 2021
# http://challenge.ai.iqiyi.com/

# In[1]:


import pandas as pd
import numpy as np
from itertools import groupby


# In[2]:

# 定义数据输入和输出目录
input_dir = "./data/"
output_dir = "./wsdm_model_data/"


# In[3]:

# 读取应用启动日志数据
launch = pd.read_csv(input_dir + "app_launch_logs.csv")
# 输出启动日期的最小值和最大值
launch.date.min(), launch.date.max()


# In[4]:

# 按用户ID分组，聚合启动日期和启动类型列表
launch_grp = launch.groupby("user_id").agg(
    launch_date=("date", list), # 获取每个用户的启动日期列表
    launch_type=("launch_type", list)# 获取每个用户的启动类型列表
).reset_index()
launch_grp


# In[5]:


#  为每个用户选择一个结束日期，用于生成样本
def choose_end_date(launch_date):
    n1, n2 = min(launch_date), max(launch_date)# 获取最早和最晚的启动日期
    # 如果最早日期和最晚日期相差超过7天，随机选择一个结束日期
    if n1 < n2 - 7:
        end_date = np.random.randint(n1, n2 - 7)
    else:
# 否则随机选择一个在100到222-7之间的日期作为结束日期
        end_date = np.random.randint(100, 222 - 7)
    return end_date
launch_grp["end_date"] = launch_grp.launch_date.apply(choose_end_date)
launch_grp


# In[6]:

# 根据结束日期生成标签，标签为结束日期后7天内的启动次数
def get_label(row):
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end+8])
    return label

launch_grp["label"] = launch_grp.apply(get_label, axis=1)
launch_grp


# In[7]:

# 输出标签的分布情况
launch_grp.label.value_counts()


# In[9]:

# 提取训练数据，包括用户ID、结束日期和标签
train = launch_grp[["user_id", "end_date", "label"]]
train


# In[10]:


# 读取测试数据
test = pd.read_csv(input_dir + "test-a.csv")
test["label"] = -1 # 测试数据的标签设置为-1
test


# In[11]:


# 合并训练数据和测试数据
data = pd.concat([train, test], ignore_index=True)
data


# 处理启动数据，为每个用户生成最新的32天启动类型序列

# In[12]:


#将测试数据追加到launch_grp中。
launch_grp = launch_grp.append(
    test.merge(launch_grp[["user_id", "launch_type", "launch_date"]], how="left", on="user_id")
)
launch_grp


# In[13]:


#获取最近32天（从[end_date-31, end_date]）的发射类型序列。
#0代表未发射，1代表发射类型为0，2代表发射类型为1。
def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end-31, end+1)]
    return seq
launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)
launch_grp


# In[15]:


data = data.merge(
    launch_grp[["user_id", "end_date", "label", "launch_seq"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
data


#数据处理

# In[16]:


# 选择 [end_date-31, end_date] 时间范围内的回放数据
playback = pd.read_csv(input_dir + "user_playback_data.csv", dtype={"item_id": str})
playback = playback.merge(data, how="inner", on="user_id")
playback = playback.loc[(playback.date >= playback.end_date-31) & (playback.date <= playback.end_date)]
playback


# In[17]:


# 将视频信息添加到回放数据中
video_data = pd.read_csv(input_dir + "video_related_data.csv", dtype=str)
playback = playback.merge(video_data[video_data.item_id.notna()], how="left", on="item_id")
playback


# In[18]:


# 使用目标编码
def target_encoding(name, df, m=1):
    df[name] = df[name].str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df = df.groupby(name).agg(
        freq=("label", "count"), 
        in_category=("label", np.mean)
    ).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df


# In[19]:


# 对 father_id 进行目标编码
df = playback.loc[(playback.label >= 0) & (playback.father_id.notna()), ["father_id", "label"]]
father_id_score = target_encoding("father_id", df)
father_id_score


# In[20]:


# 对tag_id进行目标编码
df = playback.loc[(playback.label >= 0) & (playback.tag_list.notna()), ["tag_list", "label"]]
tag_id_score = target_encoding("tag_list", df)
tag_id_score.rename({"tag_list": "tag_id"}, axis=1, inplace=True)
tag_id_score


# In[21]:


# 对cast_id进行目标编码
df = playback.loc[(playback.label >= 0) & (playback.cast.notna()), ["cast", "label"]]
cast_id_score = target_encoding("cast", df)
cast_id_score.rename({"cast": "cast_id"}, axis=1, inplace=True)
cast_id_score


# In[22]:


# 为特征工程分组播放数据
playback_grp = playback.groupby(["user_id", "end_date", "label"]).agg(
    playtime_list=("playtime", list),
    date_list=("date", list),
    duration_list=("duration", lambda x: ";".join(map(str, x))),
    father_id_list=("father_id", lambda x: ";".join(map(str, x))),
    tag_list=("tag_list", lambda x: ";".join(map(str, x))),
    cast_list=("cast", lambda x: ";".join(map(str, x)))
).reset_index()
playback_grp


# In[23]:


# 生成最近32天（[end_date-31, end_date]）的播放时间序列
# 播放量归一化公式：playtime_norm = 1/(1 + exp(3 - playtime/450))。当播放时间为3600秒时，其偏好得分几乎等于1
def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-31, row.end_date+1)]
    return seq
playback_grp["playtime_seq"] = playback_grp.apply(get_playtime_seq, axis=1)
playback_grp


# In[24]:


drn_desc = video_data.loc[video_data.duration.notna(), "duration"].astype(int)
drn_desc.min(), drn_desc.max()


# In[25]:


# 播放时长偏好是一个16维的偏好向量
# 对于一个用户，计算每个播放时长的频率
# 偏好得分 = 频率 / 最大频率
# 如果用户的时长列表全部为空，则返回空
# 空的时长偏好后续将被填充为零向量
def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i), 0)/max_, 4) for i in range(1, 17)]
        return res
    else:
        return np.nan
playback_grp["duration_prefer"] = playback_grp.duration_list.apply(get_duration_prefer)


# In[26]:


# 将所有目标编码分数添加到一个字典中
id_score = dict()
id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})

# 检查特征ID是否重复
father_id_score.shape[0]+tag_id_score.shape[0]+cast_id_score.shape[0] == len(id_score)


# In[27]:


# 对于三个特征（father_id_score、cast_score、tag_score）
# 根据频率选择前3个偏好。
# 计算前3个偏好的加权平均分数。
# 如果ID列表全部为空，则返回空。
def get_id_score(id_list):
    x = sorted(id_list.split(";"))
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
    if x_count:
        x_sort = sorted(x_count.items(), key=lambda k: -k[1])
        top_x = x_sort[:3]
        res = [(n, id_score.get(k, 0)) for k, n in top_x]
        res = sum(n*v for n, v in res) / sum(n for n, v in res)
        return res
    else:
        return np.nan


# In[28]:


playback_grp["father_id_score"] = playback_grp.father_id_list.apply(get_id_score)


# In[29]:


playback_grp["cast_id_score"] = playback_grp.cast_list.apply(get_id_score)


# In[30]:


playback_grp["tag_score"] = playback_grp.tag_list.apply(get_id_score)
playback_grp


# In[31]:


data = data.merge(
    playback_grp[["user_id", "end_date", "label", "playtime_seq", "duration_prefer", "father_id_score", "cast_id_score", "tag_score"]],
    on=["user_id", "end_date", "label"],
    how="left"
)
data


# 用户画像数据处理

# In[32]:


portrait = pd.read_csv(input_dir + "user_portrait_data.csv", dtype={"territory_code": str})
portrait = pd.merge(data[["user_id", "label"]], portrait, how="left", on="user_id")
portrait


# In[33]:


# 对于 territory_code 字段，再次应用目标编码。
df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]
territory_score = target_encoding("territory_code", df)
territory_score


# In[34]:


# 将 territory_code 的得分添加到 id_score 中。
n1 = len(id_score)
id_score.update({x[1]: x[5] for x in territory_score.itertuples()})
n1 + territory_score.shape[0] == len(id_score)


# In[35]:


#获取地区得分，保留空值
portrait["territory_score"] = portrait.territory_code.apply(lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)
portrait


# In[36]:


# 对于 device_ram 和 device_rom 的多值情况，选择第一个值
portrait["device_ram"] = portrait.device_ram.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait["device_rom"] = portrait.device_rom.apply(lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
portrait


# In[37]:


# 将肖像特征添加到数据中
data = data.merge(portrait.drop("territory_code", axis=1), how="left", on=["user_id", "label"])
data


# 交互数据处理

# In[38]:


# 仅使用交互类型偏好
# 利用所有交互数据来计算交互类型偏好
interact = pd.read_csv(input_dir + "user_interaction_data.csv")
interact.interact_type.min(), interact.interact_type.max()


# In[39]:


interact_grp = interact.groupby("user_id").agg(
    interact_type=("interact_type", list)
).reset_index()
interact_grp


# In[40]:


# 类似于时长偏好，交互类型偏好可以是一个11维向量。
def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res
interact_grp["interact_prefer"] = interact_grp.interact_type.apply(get_interact_prefer)
interact_grp


# In[41]:


data = data.merge(interact_grp[["user_id", "interact_prefer"]], on="user_id", how="left")
data


# 特征归一化并保存数据

# In[42]:


# 以下特征需要进行标准化处理
# 方法：x = (x - x的平均值) / x的标准差
norm_cols = ["father_id_score", "cast_id_score", "tag_score", 
            "device_type", "device_ram", "device_rom", "sex",
            "age", "education", "occupation_status", "territory_score"]
for col in norm_cols:
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col] - mean) / std
data


# In[43]:


# 用零向量填充空向量特征
data.fillna({
    "playtime_seq": str([0]*32),
    "duration_prefer": str([0]*16),
    "interact_prefer": str([0]*11)
}, inplace=True)
data


# In[44]:


# 用0填充空数值特征
data.fillna(0, inplace=True)
data


# In[45]:


data.loc[data.label >= 0].to_csv(output_dir + "train_data.txt", sep="\t", index=False)
data.loc[data.label < 0].to_csv(output_dir + "test_data.txt", sep="\t", index=False)

