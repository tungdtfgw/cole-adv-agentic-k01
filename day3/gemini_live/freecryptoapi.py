"""
FreeCryptoAPI Python Client Library

A Python wrapper for the FreeCryptoAPI (https://freecryptoapi.com).
Provides access to cryptocurrency market data, technical analysis,
exchange data, conversion, and historical data endpoints.
"""

import requests
from typing import Optional, Any, Dict


class FreeCryptoAPIError(Exception):
    """Exception raised when an API request fails.

    Attributes:
        status_code: HTTP status code returned by the API.
        message: Human-readable error message.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class FreeCryptoAPI:
    """Client for the FreeCryptoAPI.

    Args:
        api_key: Your FreeCryptoAPI bearer token.
        base_url: API base URL. Defaults to the production endpoint.
        timeout: Request timeout in seconds. Defaults to 30.

    Example::

        from freecryptoapi import FreeCryptoAPI

        client = FreeCryptoAPI("your_api_key")
        data = client.get_data("BTC")
        print(data)
    """

    BASE_URL = "https://api.freecryptoapi.com/v1"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a GET request and return the parsed JSON response.

        Args:
            endpoint: API endpoint path (e.g. ``/getData``).
            params: Optional query parameters. ``None`` values are stripped
                automatically so callers can pass optional arguments directly.

        Returns:
            Parsed JSON response body.

        Raises:
            FreeCryptoAPIError: If the response status code indicates an error.
        """
        url = f"{self.base_url}{endpoint}"

        # Remove keys whose value is None
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        response = self._session.get(url, params=params, timeout=self.timeout)

        if not response.ok:
            try:
                body = response.json()
                message = body.get("message") or body.get("error") or response.text
            except (ValueError, KeyError):
                message = response.text
            raise FreeCryptoAPIError(response.status_code, message)

        return response.json()

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_crypto_list(self) -> Any:
        """Retrieve the full list of supported cryptocurrencies.

        Returns:
            JSON response containing all available cryptocurrencies.

        Example::

            crypto_list = client.get_crypto_list()
        """
        return self._request("/getCryptoList")

    def get_data(self, symbol: str) -> Any:
        """Get current market data for one or more symbols.

        Args:
            symbol: Cryptocurrency symbol(s). Use ``+`` to separate multiple
                symbols (e.g. ``"BTC+ETH"``). Append ``@exchange`` to query a
                specific exchange (e.g. ``"BTC@binance"``).

        Returns:
            JSON response with market data for the requested symbol(s).

        Example::

            btc = client.get_data("BTC")
            multi = client.get_data("BTC+ETH+SOL")
            exchange = client.get_data("BTC@binance")
        """
        return self._request("/getData", {"symbol": symbol})

    def get_top(self, top: Optional[int] = None) -> Any:
        """Get top cryptocurrencies by market cap.

        Args:
            top: Number of top cryptocurrencies to return. If ``None`` the API
                default is used.

        Returns:
            JSON response with top cryptocurrency data.

        Example::

            top10 = client.get_top(10)
        """
        return self._request("/getTop", {"top": top})

    def get_data_currency(self, symbol: str, local: str) -> Any:
        """Get market data for a symbol in a specific local currency.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            local: Local fiat currency code (e.g. ``"EUR"``, ``"TRY"``).

        Returns:
            JSON response with market data in the requested currency.

        Example::

            btc_eur = client.get_data_currency("BTC", "EUR")
        """
        return self._request("/getDataCurrency", {"symbol": symbol, "local": local})

    def get_performance(self, symbol: str) -> Any:
        """Get performance metrics for a cryptocurrency.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).

        Returns:
            JSON response with performance data.

        Example::

            perf = client.get_performance("BTC")
        """
        return self._request("/getPerformance", {"symbol": symbol})

    def get_volatility(
        self, symbol: Optional[str] = None, top: Optional[int] = None
    ) -> Any:
        """Get volatility data.

        Args:
            symbol: Optional cryptocurrency symbol to filter by.
            top: Optional number of top results to return.

        Returns:
            JSON response with volatility data.

        Example::

            vol = client.get_volatility(symbol="BTC")
            top_vol = client.get_volatility(top=10)
        """
        return self._request("/getVolatility", {"symbol": symbol, "top": top})

    def get_ath_atl(
        self, symbol: Optional[str] = None, months: Optional[int] = None
    ) -> Any:
        """Get all-time high and all-time low data.

        Args:
            symbol: Optional cryptocurrency symbol to filter by.
            months: Optional number of months to look back.

        Returns:
            JSON response with ATH/ATL data.

        Example::

            ath_atl = client.get_ath_atl(symbol="BTC")
            recent = client.get_ath_atl(months=3)
        """
        return self._request("/getATHATL", {"symbol": symbol, "months": months})

    def get_fear_greed(self) -> Any:
        """Get the current Fear & Greed Index.

        Returns:
            JSON response with the Fear & Greed Index value and classification.

        Example::

            fg = client.get_fear_greed()
        """
        return self._request("/getFearGreed")

    # ------------------------------------------------------------------
    # Technical Analysis
    # ------------------------------------------------------------------

    def get_technical_analysis(self, symbol: str) -> Any:
        """Get technical analysis summary for a symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).

        Returns:
            JSON response with technical analysis indicators and signals.

        Example::

            ta = client.get_technical_analysis("BTC")
        """
        return self._request("/getTechnicalAnalysis", {"symbol": symbol})

    def get_breakouts(self, symbol: Optional[str] = None) -> Any:
        """Get breakout signals.

        Args:
            symbol: Optional cryptocurrency symbol to filter by.

        Returns:
            JSON response with breakout data.

        Example::

            breakouts = client.get_breakouts()
            btc_breakouts = client.get_breakouts(symbol="BTC")
        """
        return self._request("/getBreakouts", {"symbol": symbol})

    def get_correlation(self, symbols: str, days: Optional[int] = None) -> Any:
        """Get correlation data between symbols.

        Args:
            symbols: Symbols to correlate (e.g. ``"BTC+ETH"``).
            days: Optional number of days to calculate correlation over.

        Returns:
            JSON response with correlation matrix/data.

        Example::

            corr = client.get_correlation("BTC+ETH", days=30)
        """
        return self._request("/getCorrelation", {"symbols": symbols, "days": days})

    def get_support_resistance(
        self, symbol: str, period: Optional[int] = None
    ) -> Any:
        """Get support and resistance levels for a symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            period: Optional period for calculation.

        Returns:
            JSON response with support and resistance levels.

        Example::

            sr = client.get_support_resistance("BTC")
            sr_custom = client.get_support_resistance("BTC", period=14)
        """
        return self._request(
            "/getSupportResistance", {"symbol": symbol, "period": period}
        )

    def get_ma_ribbon(self, symbol: str, days: Optional[int] = None) -> Any:
        """Get moving average ribbon data for a symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            days: Optional number of days for the ribbon calculation.

        Returns:
            JSON response with moving average ribbon data.

        Example::

            ribbon = client.get_ma_ribbon("BTC")
            ribbon_60 = client.get_ma_ribbon("BTC", days=60)
        """
        return self._request("/getMARibbon", {"symbol": symbol, "days": days})

    def get_bollinger(
        self,
        symbol: str,
        days: Optional[int] = None,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> Any:
        """Get Bollinger Bands data for a symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            days: Optional number of days of data.
            period: Optional Bollinger Band period (default is typically 20).
            std_dev: Optional standard deviation multiplier (default is
                typically 2.0).

        Returns:
            JSON response with Bollinger Bands data.

        Example::

            bb = client.get_bollinger("BTC")
            bb_custom = client.get_bollinger("BTC", days=30, period=20, std_dev=2.5)
        """
        return self._request(
            "/getBollinger",
            {
                "symbol": symbol,
                "days": days,
                "period": period,
                "std_dev": std_dev,
            },
        )

    # ------------------------------------------------------------------
    # Exchange Data
    # ------------------------------------------------------------------

    def get_exchange(self, exchange: str) -> Any:
        """Get data for a specific exchange.

        Args:
            exchange: Exchange identifier (e.g. ``"binance"``, ``"coinbase"``).

        Returns:
            JSON response with exchange data.

        Example::

            binance = client.get_exchange("binance")
        """
        return self._request("/getExchange", {"exchange": exchange})

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def get_conversion(
        self, from_currency: str, to_currency: str, amount: float
    ) -> Any:
        """Convert between currencies.

        Args:
            from_currency: Source currency (e.g. ``"BTC"``).
            to_currency: Target currency (e.g. ``"USD"``).
            amount: Amount to convert.

        Returns:
            JSON response with conversion result.

        Example::

            result = client.get_conversion("BTC", "USD", 1.5)
        """
        return self._request(
            "/getConversion",
            {"from": from_currency, "to": to_currency, "amount": amount},
        )

    # ------------------------------------------------------------------
    # Historical Data
    # ------------------------------------------------------------------

    def get_history(self, symbol: str, days: int) -> Any:
        """Get historical price data for a symbol.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            days: Number of days of historical data.

        Returns:
            JSON response with historical price data.

        Example::

            history = client.get_history("BTC", 30)
        """
        return self._request("/getHistory", {"symbol": symbol, "days": days})

    def get_timeframe(self, symbol: str, start: str, end: str) -> Any:
        """Get price data for a specific time range.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.

        Returns:
            JSON response with price data for the specified timeframe.

        Example::

            tf = client.get_timeframe("BTC", "2024-01-01", "2024-01-31")
        """
        return self._request(
            "/getTimeframe", {"symbol": symbol, "start": start, "end": end}
        )

    def get_ohlc(
        self,
        symbol: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Any:
        """Get OHLC (Open/High/Low/Close) candlestick data.

        Args:
            symbol: Cryptocurrency symbol (e.g. ``"BTC"``).
            days: Optional number of days of data.
            start_date: Optional start date in ``YYYY-MM-DD`` format.
            end_date: Optional end date in ``YYYY-MM-DD`` format.

        Returns:
            JSON response with OHLC data.

        Example::

            ohlc = client.get_ohlc("BTC", days=7)
            ohlc_range = client.get_ohlc("BTC", start_date="2024-01-01", end_date="2024-01-31")
        """
        return self._request(
            "/getOHLC",
            {
                "symbol": symbol,
                "days": days,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
