import time

from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.gateway import (
    AuthType,
    GrazieAgent,
    GrazieApiGatewayClient,
    GrazieApiGatewayUrls,
    RequestFailedException,
)
from grazie.api.client.llm_parameters import LLMParameters
from grazie.api.client.parameters import Parameters
from grazie.api.client.profiles import LLMProfile, Profile

from commit_style.llm.base import LLMClient


class GrazieLLMClient(LLMClient):
    """Wraps JetBrains' internal Grazie gateway for chat completion.

    Auth: pass `token` explicitly, or leave it unset and export one of
    GRAZIE_JWT_TOKEN / GRAZIE_USER_JWT_TOKEN / GRAZIE_SERVICE_JWT_TOKEN — the underlying
    GrazieApiGatewayClient reads those itself.
    """

    _MAX_RETRIES = 3
    _BACKOFF_SECONDS = 2.0

    def __init__(
        self,
        model_name: str,
        token: str | None = None,
        temperature: float = 0.0,
        auth_type: AuthType = AuthType.USER,
        prompt_id: str = "commit-with-style",
    ):
        self._client = GrazieApiGatewayClient(
            grazie_agent=GrazieAgent(name="commit-with-style", version="dev"),
            url=GrazieApiGatewayUrls.PRODUCTION,
            auth_type=auth_type,
            grazie_jwt_token=token,
        )
        self._profile = self._resolve_profile(model_name)
        self._temperature = temperature
        self._prompt_id = prompt_id

    @staticmethod
    def _resolve_profile(model_name: str) -> LLMProfile:
        try:
            return Profile.get_by_name(model_name)
        except ValueError:
            return LLMProfile(name=model_name)

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        chat = ChatPrompt().add_system(system).add_user(user)
        parameters = {
            LLMParameters.Temperature: Parameters.FloatValue(self._temperature),
            LLMParameters.Length: Parameters.IntValue(max_tokens),
        }
        attempt = 0
        start = time.time()
        while True:
            try:
                response = self._client.chat(
                    chat=chat, profile=self._profile, prompt_id=self._prompt_id, parameters=parameters
                )
                self.last_usage = {
                    # Grazie's ChatResponse doesn't expose raw token counts, only quota
                    # credits spent — leave the token fields None rather than estimate.
                    "input_tokens": None,
                    "output_tokens": None,
                    "latency_s": time.time() - start,
                }
                return response.content
            except RequestFailedException as e:
                message = str(e).lower()
                # Some models (e.g. gpt-5, a reasoning model) reject sampling params like
                # temperature entirely — drop it and retry immediately, no backoff needed
                # since this is a deterministic fix, not a transient failure.
                if "not supported" in message and "temperature" in message and LLMParameters.Temperature in parameters:
                    del parameters[LLMParameters.Temperature]
                    continue
                # Anything else (e.g. "Server error: Unexpected error") is treated as a
                # transient gateway hiccup — retry with exponential backoff before giving up,
                # so a long batch run isn't aborted by one flaky request.
                attempt += 1
                if attempt > self._MAX_RETRIES:
                    raise
                wait_seconds = self._BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(
                    f"[GrazieLLMClient] transient error ({e}); retrying in {wait_seconds:.0f}s "
                    f"(attempt {attempt}/{self._MAX_RETRIES})"
                )
                time.sleep(wait_seconds)
