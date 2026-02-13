from llamafactory.data.formatter import EmptyFormatter


# 模拟环境中的 FunctionCall 类
class FunctionCall:
    pass


# 演示代码
def demo_empty_formatter():
    print("=== 示例 1: 正常的静态标识符处理 ===")
    # 假设这是某个模型 User 角色的固定前缀，不含动态内容
    # {"token": "..."} 在 LLaMA-Factory 中代表这是一个特殊 Token，不应该被分词器二次切分
    user_header = EmptyFormatter(
        slots=[
            {"token": "<|start_header_id|>"},
            "user",
            {"token": "<|end_header_id|>\n\n"},
        ]
    )

    # 调用 apply
    result = user_header.apply(content="这一段会被忽略")
    print(f"填充后的结果: {result}")
    # 输出结果只包含 slots 定义的静态内容

    print("\n=== 示例 2: 触发逻辑校验错误 ===")
    try:
        # 错误演示：如果把动态槽位误传给 EmptyFormatter
        invalid_formatter = EmptyFormatter(slots=["User: {{content}}"])
    except ValueError as e:
        print(f"校验成功截获错误: {e}")


# 模拟运行
if __name__ == "__main__":
    demo_empty_formatter()

"""
$ python src/llamafactory/data/formatter_test1.py
=== 示例 1: 正常的静态标识符处理 ===
填充后的结果: [{'token': '<|start_header_id|>'}, 'user', {'token': '<|end_header_id|>\n\n'}]

=== 示例 2: 触发逻辑校验错误 ===
校验成功截获错误: Empty formatter should not contain any placeholder.
$
"""
