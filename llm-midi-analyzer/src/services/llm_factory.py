class LLMFactory:
    @staticmethod
    def get_llm(local=True):
        if local:
            from services.unsloth_engine import LLMEngine
            return LLMEngine()
        else:
            # Placeholder for Gemini API
            class RemoteLLM:
                def generate(self, prompt):
                    return "Remote call simulated"
            return RemoteLLM()
