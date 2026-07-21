import os

from haystack.utils.auth import Secret
from voyageai import AsyncClient, Client


class VoyageClientMixin:
    """Shared lifecycle management for Voyage AI sync and async clients.

    Provides lazy initialization via ``client`` / ``async_client`` properties
    that auto-heal by calling ``warm_up()`` when the backing field is ``None``.
    Subclasses must call ``_init_client_lifecycle`` in their ``__init__``.
    """

    _client: Client | None
    _async_client: AsyncClient | None
    _timeout: int
    _max_retries: int

    def _init_client_lifecycle(
        self,
        api_key: Secret,
        timeout: int | None,
        max_retries: int | None,
        default_timeout: int = 30,
        default_max_retries: int = 5,
    ) -> None:
        self.api_key = api_key
        if timeout is None:
            timeout = int(os.environ.get("VOYAGE_TIMEOUT", str(default_timeout)))
        if max_retries is None:
            max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", str(default_max_retries)))
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None
        self._async_client = None

    @property
    def client(self) -> Client:
        """Get the synchronous Voyage AI client, initializing it on first access."""
        if self._client is None:
            self.warm_up()
        return self._client  # ty: ignore

    @property
    def async_client(self) -> AsyncClient:
        """Get the asynchronous Voyage AI client, initializing it on first access."""
        if self._async_client is None:
            self.warm_up()
        return self._async_client  # ty: ignore

    def warm_up(self) -> None:
        """Initialize the Voyage AI clients if they haven't been initialized yet."""
        if self._client is not None and self._async_client is not None:
            return
        api_key = self.api_key.resolve_value()
        self._client = Client(api_key=api_key, max_retries=self._max_retries, timeout=self._timeout)
        self._async_client = AsyncClient(api_key=api_key, max_retries=self._max_retries, timeout=self._timeout)
