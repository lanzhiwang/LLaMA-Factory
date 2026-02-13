from llamafactory.data.formatter import StringFormatter

# 1. 构造一个 StringFormatter 实例
# 假设 Llama-3 的格式是: <|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>
llama3_user_formatter = StringFormatter(
    slots=[
        # 这是一个特殊 Token 标记(以字典形式存在, 后续会被单独处理成 ID)
        {"token": "<|start_header_id|>"},
        "user",
        {"token": "<|end_header_id|>"},
        "\n\n",
        # 这是一个占位符, 稍后将被用户输入的文本替换
        "{{content}}",
        # 结尾标记
        {"token": "<|eot_id|>"},
    ]
)

# 2. 模拟实际调用数据
user_input = "请问如何学习 Python?"

# 3. 调用 apply 方法进行填充
try:
    final_slots = llama3_user_formatter.apply(content=user_input)

    # 4. 演示输出结果
    print("--- 填充后的 Slots 结果 ---")
    for i, slot in enumerate(final_slots):
        print(f"槽位 [{i}] 类型: {type(slot).__name__:<5} 内容: {slot}")

except Exception as e:
    print(f"执行失败: {e}")

# --- 进阶演示: 错误校验 ---
print("\n--- 错误校验演示 ---")
try:
    # 故意创建一个没有占位符的 StringFormatter
    invalid_formatter = StringFormatter(slots=["Hello World"])
except ValueError as e:
    print(f"校验成功捕捉到错误: {e}")

"""
$ python src/llamafactory/data/formatter_test2.py
--- 填充后的 Slots 结果 ---
槽位 [0] 类型: dict  内容: {'token': '<|start_header_id|>'}
槽位 [1] 类型: str   内容: user
槽位 [2] 类型: dict  内容: {'token': '<|end_header_id|>'}
槽位 [3] 类型: str   内容:


槽位 [4] 类型: str   内容: 请问如何学习 Python?
槽位 [5] 类型: dict  内容: {'token': '<|eot_id|>'}

--- 错误校验演示 ---
校验成功捕捉到错误: A placeholder is required in the string formatter.
$
"""
