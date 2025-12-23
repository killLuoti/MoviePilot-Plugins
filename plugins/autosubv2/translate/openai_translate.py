import time
from typing import List, Union

import httpx
from openai import OpenAI
from cacheout import Cache

# 会话缓存
OpenAISessionCache = Cache(maxsize=100, ttl=3600, timer=time.time, default=None)


class OpenAi:
    _api_key: str = None
    _api_url: str = None
    _model: str = "gpt-4.1-mini"
    _client: OpenAI = None

    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
        proxy: dict = None,
        model: str = None,
        compatible: bool = False,
    ):
        self._api_key = api_key
        self._api_url = api_url

        # base_url 处理（兼容第三方中转）
        base_url = self._api_url if compatible else f"{self._api_url}/v1"

        # 代理
        http_client = None
        if proxy and proxy.get("https"):
            http_client = httpx.Client(proxy=proxy.get("https"))

        # 初始化 OpenAI 客户端
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=base_url,
            http_client=http_client,
        )

        if model:
            self._model = model

    @staticmethod
    def __save_session(session_id: str, message: str):
        session = OpenAISessionCache.get(session_id)
        if session:
            session.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )
            OpenAISessionCache.set(session_id, session)

    @staticmethod
    def __get_session(session_id: str, message: str) -> List[dict]:
        session = OpenAISessionCache.get(session_id)
        if session:
            session.append(
                {
                    "role": "user",
                    "content": message,
                }
            )
        else:
            session = [
                {
                    "role": "system",
                    "content": "请在接下来的对话中使用中文回复，并且内容尽可能详细。",
                },
                {
                    "role": "user",
                    "content": message,
                },
            ]
            OpenAISessionCache.set(session_id, session)
        return session

    def __get_model(
        self,
        message: Union[str, List[dict]],
        prompt: str = None,
        **kwargs,
    ):
        """
        使用 OpenAI Responses API 获取模型响应（openai>=1.0.0）
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ]
            else:
                message = [
                    {"role": "user", "content": message},
                ]

        response = self._client.responses.create(
            model=self._model,
            input=message,
            **kwargs,
        )
        return response

    @staticmethod
    def __clear_session(session_id: str):
        if OpenAISessionCache.get(session_id):
            OpenAISessionCache.delete(session_id)

    def translate_to_zh(self, text: str, context: str = None):
        system_prompt = """您是一位专业字幕翻译专家，请严格遵循以下规则：
1. 将原文精准翻译为简体中文，保持原文本意
2. 使用自然的口语化表达，符合中文观影习惯
3. 结合上下文语境，人物称谓、专业术语、情感语气在上下文中保持连贯
4. 按行翻译待译内容。翻译结果不要包括上下文。
5. 输出内容必须仅包括译文。不要输出任何开场白、解释说明或总结"""

        user_prompt = (
            f"翻译上下文：\n{context}\n\n需要翻译的内容：\n{text}"
            if context
            else f"请翻译：\n{text}"
        )

        try:
            completion = self.__get_model(
                prompt=system_prompt,
                message=user_prompt,
                temperature=0.2,
                top_p=0.9,
            )

            # Responses API 正确读取方式
            result = completion.output_text.strip()
            return True, result

        except Exception as e:
            return False, f"翻译发生错误：{str(e)}"
