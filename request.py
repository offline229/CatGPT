import requests
import json
import time

# API 相关配置
CHAT_URL = "https://api.coze.cn/v3/chat"
MESSAGE_LIST_URL = "https://api.coze.cn/v3/chat/message/list"
HEADERS = {
    "Authorization": "Bearer pat_vXDMjQzI98QKO9gUUN5nEU8YHZeu4v7heWMmjH4LFXzjpnA1o5Bmwvr2JCFVqond",
    "Content-Type": "application/json"
}

# 发送对话请求
def start_chat():
    data = {
        "bot_id": "7486346843451293708",
        "user_id": "123123***",
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {
                "role": "user",
                "content": "早上好",
                "content_type": "text"
            }
        ]
    }

    response = requests.post(CHAT_URL, headers=HEADERS, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        print("对话请求返回:", json.dumps(response_data, indent=2, ensure_ascii=False))

        # 检查 API 响应状态
        if response_data.get("code") == 0:
            chat_id = response_data.get("data", {}).get("id")
            conversation_id = response_data.get("data", {}).get("conversation_id")

            if chat_id and conversation_id:
                print(f"成功获取 chat_id: {chat_id}, conversation_id: {conversation_id}")
                return chat_id, conversation_id
            else:
                print("Error: 未找到 chat_id 或 conversation_id")
                return None, None
        else:
            print(f"API 调用失败，错误信息: {response_data.get('msg')}")
            return None, None
    else:
        print(f"请求失败，HTTP 状态码: {response.status_code}")
        return None, None

# 查询对话消息
def fetch_chat_messages(chat_id, conversation_id):
    params = {
        "chat_id": chat_id,
        "conversation_id": conversation_id
    }
    
    # 等待 5 秒，确保消息处理完成
    time.sleep(1)

    response = requests.get(MESSAGE_LIST_URL, headers=HEADERS, params=params)

    if response.status_code == 200:
        response_data = response.json()
        print("查询消息返回:", json.dumps(response_data, indent=2, ensure_ascii=False))

        # 解析 API 返回的消息数据
        if response_data.get("code") == 0:
            messages = response_data.get("data", [])  # 这里修正，data 直接是列表

            print("\n对话消息记录：")
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "（无内容）")
                print(f"{role}: {content}")
        else:
            print(f"查询失败，错误信息: {response_data.get('msg')}")
    else:
        print(f"请求失败，HTTP 状态码: {response.status_code}")

# 执行完整流程
if __name__ == "__main__":
    chat_id, conversation_id = start_chat()
    if chat_id and conversation_id:
        fetch_chat_messages(chat_id, conversation_id)
