from langchain_community.chat_message_histories import ChatMessageHistory
history = ChatMessageHistory()

# 写入两条记录
history.add_user_message('你好')
history.add_ai_message('hello')
print(history)
# 读取并打印
for i,m in enumerate(history.messages):
    print(f'{i}: {m.content}')