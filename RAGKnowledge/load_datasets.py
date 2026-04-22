from datasets import Dataset
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), "dataset", "Chart-MRAG", "data", "data-00000-of-00001.arrow")
dataset = Dataset.from_file(data_path)

# Access a sample
sample = dataset[10]

# Access different fields
question = sample['query']
answer = sample['gt_answer']
chart = sample['gt_chart']  # Image data
text = sample['gt_text']

# 展示文件类型
print(type(question))
print(type(answer))
print(type(chart))
print(type(text))