"""Crypto data tools using FreeCryptoAPI for Gemini function calling."""

from freecryptoapi import FreeCryptoAPI


def get_crypto_tool_declarations():
    """Return Gemini-compatible function declarations for crypto tools."""
    return [
        {
            "name": "get_crypto_price",
            "description": "Get live price and market data for one or more cryptocurrencies. Use '+' to separate multiple symbols (e.g. 'BTC+ETH+SOL').",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": "Crypto symbol(s), e.g. 'BTC', 'ETH', 'BTC+ETH+SOL'",
                    }
                },
                "required": ["symbols"],
            },
        },
        {
            "name": "convert_crypto",
            "description": "Convert an amount from one cryptocurrency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_symbol": {
                        "type": "string",
                        "description": "Source crypto symbol, e.g. 'BTC'",
                    },
                    "to_symbol": {
                        "type": "string",
                        "description": "Target crypto symbol, e.g. 'ETH'",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to convert",
                    },
                },
                "required": ["from_symbol", "to_symbol", "amount"],
            },
        },
        {
            "name": "get_technical_analysis",
            "description": "Get technical analysis indicators (RSI, MACD, etc.) for a cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto symbol, e.g. 'BTC'",
                    }
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_bollinger_bands",
            "description": "Get Bollinger Bands data for a cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto symbol, e.g. 'ETH'",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days for calculation (default 90)",
                    },
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_fear_greed_index",
            "description": "Get the current Fear & Greed Index for the crypto market. No parameters needed.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


class CryptoToolExecutor:
    """Executes crypto tool calls using FreeCryptoAPI."""

    def __init__(self, api_key: str):
        self.api = FreeCryptoAPI(api_key)

    def execute(self, function_name: str, args: dict) -> dict:
        """Execute a crypto tool function and return result as dict."""
        try:
            if function_name == "get_crypto_price":
                result = self.api.get_data(args["symbols"])
                return {"result": str(result)}

            elif function_name == "convert_crypto":
                result = self.api.get_conversion(
                    args["from_symbol"],
                    args["to_symbol"],
                    args.get("amount", 1),
                )
                return {"result": str(result)}

            elif function_name == "get_technical_analysis":
                result = self.api.get_technical_analysis(args["symbol"])
                return {"result": str(result)}

            elif function_name == "get_bollinger_bands":
                days = args.get("days", 90)
                result = self.api.get_bollinger(args["symbol"], days=days)
                return {"result": str(result)}

            elif function_name == "get_fear_greed_index":
                result = self.api.get_fear_greed()
                return {"result": str(result)}

            else:
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            return {"error": str(e)}
