import core
import copy

class Context:
    # special message type (not intended to be added to context) that
    # will cause context.get() to cut off messages before this cutoff point
    SUMMARIZATION_CUTOFF = {"signal": "SUMMARIZATION_CUTOFF"}

    def __init__(self, channel):
        self.channel = channel

        # UI-agnostic chat history system - save/load context windows from save file!
        self.chat = core.chat.Chat(self.channel)

    async def get(self, system_prompt=True, end_prompt=True, history=True, prevent_recursion=False):
        """
        builds the full context window using system prompt + message history + end prompt
        to the API, we send this full context.

        to frontend channels, we send only the message history part of the context (context.chat.get()),
        without the system prompt and without the modifications we do to it such as the endprompt.

        context must ALWAYS follow this strict turn order: system->user->assistant->user->assistant->user->...
        """

        if not self.channel.manager.API.connected:
            return None

        max_messages = int(core.config.get("api").get("max_messages", 200))
        max_tokens = int(core.config.get("api").get("max_context", 8192))
        system_role = "system" if not self.channel.manager.API.supports_developer_role else "developer"
        dev_role = "developer" if self.channel.manager.API.supports_developer_role else "user"

        system_msg = []
        if system_prompt:
            content = await self.channel.manager.get_system_prompt()
            if content:
                system_msg = [{"role": system_role, "content": content}]

        messages = []
        if history:
            messages = copy.deepcopy(await self.chat.get())

            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("signal") == "SUMMARIZATION_CUTOFF":
                    messages = [{"role": "user", "content": "Summarize our chat so far."}] + messages[i + 1:]
                    break

            messages = [msg for msg in messages if not msg.get("ghost") and not msg.get("signal")]
            messages = [
                msg for msg in messages
                if not (msg.get("role") == "assistant" and not msg.get("content") and not msg.get("tool_calls"))
            ]

            if not core.config.get("model", "keep_reasoning_in_context"):
                messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]

            if len(messages) > max_messages:
                messages = messages[-max_messages:]

            if messages:
                for i in range(len(messages) - 1):
                    msg = messages[i]
                    if msg.get("role") in ("tool", "tool_calls"):
                        continue

                    content = msg.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            part for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        if text_parts:
                            msg["content"] = text_parts
                        else:
                            msg["content"] = "[multimedia content]"

            if messages:
                enforced_messages = []
                for msg in messages:
                    if enforced_messages:
                        last_role = enforced_messages[-1].get("role")
                        current_role = msg.get("role")

                        if last_role == "assistant" and current_role == "assistant":
                            enforced_messages.append({"role": "user", "content": " "})
                        elif last_role == "user" and current_role == "user":
                            enforced_messages.append({"role": "assistant", "content": " "})

                    enforced_messages.append(msg)

                messages = enforced_messages

        end_msg = []
        if end_prompt:
            histend = await self.channel.manager.get_end_prompt(prevent_recursion=prevent_recursion)
            if histend:
                end_msg = [{"role": dev_role, "content": histend}]

        full_context = system_msg + messages + end_msg
        current_tokens = await self.chat.count_tokens(full_context)
        effective_max_tokens = int(max_tokens * 0.95)

        if current_tokens > effective_max_tokens and messages:
            lo, hi = 0, len(messages)
            best_trim = len(messages)

            while lo <= hi:
                mid = (lo + hi) // 2
                trimmed = messages[mid:]
                candidate_context = system_msg + trimmed + end_msg
                tokens = await self.chat.count_tokens(candidate_context)

                if tokens <= effective_max_tokens:
                    best_trim = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            messages = messages[best_trim:]
            full_context = system_msg + messages + end_msg
            current_tokens = await self.chat.count_tokens(full_context)

        if current_tokens > max_tokens:
            await self.channel.announce(
                "Your request exceeds the maximum token limit. Please send a smaller message!",
                "error"
            )
            return None

        return full_context

    async def get_size(self):
        message_history = await self.get(system_prompt=False)
        sysprompt = await self.channel.manager.get_system_prompt()
        histend = await self.channel.manager.get_end_prompt()
        
        sysprompt_size_tokens = await self.chat.count_tokens([{"role": "system", "content": sysprompt}])
        sysprompt_size_words = len(str(sysprompt).split())
        
        message_hist_size_tokens = await self.chat.count_tokens(await self.chat.get())
        message_hist_size_words = len(str(message_history).split())
        
        histend_size_tokens = await self.chat.count_tokens([{"role": "user", "content": histend}]) if histend else 0
        histend_size_words = len(str(histend).split()) if histend else 0

        combined_size_words = message_hist_size_words + sysprompt_size_words + histend_size_words
        token_usage = self._get_cached_token_usage()
        if not token_usage:
            token_usage = await self.chat.count_tokens(await self.get(system_prompt=True))

        return {
            "system prompt size": f"{sysprompt_size_tokens} tokens | {sysprompt_size_words} words",
            "message history size": f"{message_hist_size_tokens} tokens | {message_hist_size_words} words",
            "end prompt size": f"{histend_size_tokens} tokens | {histend_size_words} words",
            "total size": f"{token_usage} tokens | {combined_size_words} words",
        }

    def _get_cached_token_usage(self):
        try:
            current_index = self.chat.current
            if current_index is None:
                return 0
            return int(self.chat.data[current_index].get("token_usage", 0) or 0)
        except Exception:
            return 0

    async def get_token_usage(self):
        """
        Return cached token usage for lightweight UI/status polling.

        This deliberately avoids rebuilding the full prompt on page load, chat
        switch, or periodic status checks. Accurate trimming still happens inside
        get() immediately before sending a request to the model, and API-provided
        usage updates this cache after real responses.
        """
        max_tokens = core.config.get("api").get("max_context", 8192)

        return {
            "current": self._get_cached_token_usage(),
            "max": max_tokens,
            "source": "cached"
        }
