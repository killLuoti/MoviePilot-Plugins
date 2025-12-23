import time
import random
from typing import List, Union

import httpx
from openai import OpenAI
from cacheout import Cache

# 会话缓存
OpenAISessionCache = Cache(maxsize=100, ttl=3600, timer=time.time, default=None)


class OpenAi:
    _api_key: str = None
    _api_url: str = None
    _model: str = "gpt-3.5-turbo"

    def __init__(self, api_key: str = None, api_url: str = None, proxy: dict = None, model: str = None,
                 compatible: bool = False):
        """
        初始化 OpenAI 客户端 (适配 v1.0.0+)
        """
        self._api_key = api_key
        self._api_url = api_url
        if model:
            self._model = model

        # 处理 Base URL
        # v1 客户端要求 base_url 必须以 /v1 结尾（除非是兼容模式）
        base_url = None
        if self._api_url:
            base_url = self._api_url if compatible else self._api_url.rstrip("/") + "/v1"

        # 处理代理
        # OpenAI v1 不再支持 openai.proxy，必须通过 httpx 客户端注入
        http_client = None
        if proxy and proxy.get("https"):
            http_client = httpx.Client(proxies=proxy.get("https"))

        # 实例化客户端对象，避免使用全局 openai 变量
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=base_url,
            http_client=http_client
        )

    @staticmethod
    def __save_session(session_id: str, message: str):
        """
        保存会话
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "assistant",
                "content": message
            })
            OpenAISessionCache.set(session_id, seasion)

    @staticmethod
    def __get_session(session_id: str, message: str) -> List[dict]:
        """
        获取会话
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "user",
                "content": message
            })
        else:
            seasion = [
                {
                    "role": "system",
                    "content": "请在接下来的对话中请使用中文回复，并且内容尽可能详细。"
                },
                {
                    "role": "user",
                    "content": message
                }]
            OpenAISessionCache.set(session_id, seasion)
        return seasion

    def __get_model(self, message: Union[str, List[dict]],
                    prompt: str = None,
                    user: str = "MoviePilot",
                    **kwargs):
        """
        获取模型响应
        """
        if not isinstance(message, list):
            if prompt:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ]
            else:
                messages = [
                    {"role": "user", "content": message}
                ]
        else:
            messages = message
        
        # 核心修复：必须通过 self.client 调用，不能使用 openai.ChatCompletion
        return self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            user=user,
            **kwargs
        )

    @staticmethod
    def __clear_session(session_id: str):
        """
        清除会话
        """
        if OpenAISessionCache.get(session_id):
            OpenAISessionCache.delete(session_id)

    def translate_to_zh(self, text: str, context: str = None, max_retries: int = 3):
        """
        翻译为中文
        """
        system_prompt = """您是一位专业字幕翻译专家，请严格遵循以下规则：
1. 将原文精准翻译为简体中文，保持原文本意
2. 使用自然的口语化表达，符合中文观影习惯
3. 结合上下文语境，人物称谓、专业术语、情感语气在上下文中保持连贯
4. 按行翻译待译内容。翻译结果不要包括上下文。
5. 输出内容必须仅包括译文。不要输出任何开场白，解释说明或总结"""
        
        user_prompt = f"翻译上下文：\n{context}\n\n需要翻译的内容：\n{text}" if context else f"请翻译：\n{text}"
        
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                # 确保调用内部包装的新版请求方法
                completion = self.__get_model(
                    prompt=system_prompt,
                    message=user_prompt,
                    temperature=0.2,
                    top_p=0.9
                )
                # v1 响应对象通过属性访问
                result = completion.choices[0].message.content.strip()
                return True, result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    base_delay = 2 ** attempt
                    jitter = random.uniform(0.1, 0.9)
                    sleep_time = base_delay + jitter
                    print(f"翻译请求失败 (第{attempt + 1}次尝试)：{last_error}，{sleep_time:.1f}秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print(f"翻译请求失败 (已重试{max_retries}次)：{last_error}")
                    return False, f"{last_error}"
