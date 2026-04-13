from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

def send_email(to: str, subject: str, body: str):
    """发送一封邮件"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... 邮件发送逻辑

    return f"Email sent to {to}"

llm = ChatTongyi(model="qwen3-max")



agent = create_agent(
    llm,
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)