"""Authenticated Kalshi trading client: balance, positions, order book, IOC limit orders.

Only these endpoints are reachable; anything else raises. Orders always carry a
client_order_id so a retried POST cannot create a duplicate (Kalshi rejects a reused id
with 409).
"""

import time
import uuid
from dataclasses import dataclass

import httpx

from executor.auth import auth_headers

READ_RETRY_DELAYS_S = (0.5, 1.0, 2.0)
RETRY_STATUS = {429, 500, 502, 503, 504}

ALLOWED = {
    ("GET", "/portfolio/balance"),
    ("GET", "/portfolio/positions"),
    ("GET", "/portfolio/fills"),
    ("GET", "/portfolio/settlements"),
    ("POST", "/portfolio/events/orders"),
}


class TradingError(RuntimeError):
    def __init__(self, status: int | None, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status


@dataclass(frozen=True)
class OrderAck:
    order_id: str
    client_order_id: str
    fill_count: float
    remaining_count: float
    average_fill_price: float | None
    average_fee_paid: float | None


def yes_book_side(outcome: str) -> str:
    """Kalshi V2 quotes from the YES leg: buy YES = bid; buy NO = ask (sell YES)."""
    if outcome not in ("yes", "no"):
        raise ValueError(outcome)
    return "bid" if outcome == "yes" else "ask"


def yes_leg_price(outcome: str, price: float) -> float:
    """Price we pay for ``outcome`` -> the YES-leg price the V2 endpoint expects."""
    return price if outcome == "yes" else round(1 - price, 4)


class KalshiTradingClient:
    def __init__(self, base_url: str, key_id: str, key, *, timeout_s: float = 15.0,
                 transport: httpx.BaseTransport | None = None, clock=time.time,
                 sleep=time.sleep):
        if not key_id:
            raise ValueError("missing API key id")
        self.base_url = base_url.rstrip("/")
        self._key_id, self._key, self._clock = key_id, key, clock
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self._sleep = sleep

    def close(self):
        self._http.close()

    def _call(self, method: str, path: str, *, params=None, json=None):
        if (method, path) not in ALLOWED:
            raise PermissionError(f"executor endpoint not allowed: {method} {path}")
        url = self.base_url + path
        headers = auth_headers(self._key_id, self._key, method, url, int(self._clock() * 1000))
        if json is not None:
            headers["Content-Type"] = "application/json"
        resp = self._http.request(method, url, params=params, json=json, headers=headers)
        # Reads are retried on rate limits / server errors. Orders never are: an order whose
        # response was lost may still exist, and the next cycle reconciles from positions.
        for delay in (READ_RETRY_DELAYS_S if method == "GET" else ()):
            if resp.status_code not in RETRY_STATUS:
                break
            self._sleep(delay)
            headers = auth_headers(self._key_id, self._key, method, url,
                                   int(self._clock() * 1000))
            resp = self._http.request(method, url, params=params, headers=headers)
        if resp.status_code not in (200, 201):
            raise TradingError(resp.status_code, resp.text[:500])
        return resp.json()

    def balance_usd(self) -> tuple[float, float]:
        """(available cash, portfolio value) in dollars."""
        d = self._call("GET", "/portfolio/balance")
        return float(d["balance_dollars"]), float(d.get("portfolio_value", 0)) / 100.0

    def positions(self) -> list[dict]:
        out, cursor = [], None
        while True:
            params = {"limit": 1000, "count_filter": "position"}
            if cursor:
                params["cursor"] = cursor
            d = self._call("GET", "/portfolio/positions", params=params)
            for p in d.get("market_positions") or []:
                qty = float(p["position_fp"])
                out.append({"ticker": p["ticker"], "side": "yes" if qty > 0 else "no",
                            "qty": abs(qty), "exposure_usd": float(p["market_exposure_dollars"]),
                            "fees_usd": float(p["fees_paid_dollars"])})
            cursor = d.get("cursor")
            if not cursor:
                return out

    def _pages(self, path: str, key: str, params: dict) -> list[dict]:
        out, cursor = [], None
        while True:
            q = {**params, "limit": 1000}
            if cursor:
                q["cursor"] = cursor
            d = self._call("GET", path, params=q)
            out += d.get(key) or []
            cursor = d.get("cursor")
            if not cursor:
                return out

    def fills(self, min_ts: int | None = None) -> list[dict]:
        """Raw fill records (every matched trade), newest first."""
        return self._pages("/portfolio/fills", "fills",
                           {} if min_ts is None else {"min_ts": min_ts})

    def settlements(self, min_ts: int | None = None) -> list[dict]:
        """Raw settlement records for markets that resolved while we held a position."""
        return self._pages("/portfolio/settlements", "settlements",
                           {} if min_ts is None else {"min_ts": min_ts})

    def place_ioc_buy(self, ticker: str, outcome: str, count: int, max_price: float,
                      client_order_id: str | None = None) -> OrderAck:
        """Immediate-or-cancel limit buy: fills now at <= max_price or not at all."""
        if count < 1 or not 0.01 <= max_price <= 0.99:
            raise ValueError("count must be >= 1 and price within [0.01, 0.99]")
        cid = client_order_id or str(uuid.uuid4())
        body = {
            "ticker": ticker, "client_order_id": cid, "side": yes_book_side(outcome),
            "count": str(int(count)), "price": f"{yes_leg_price(outcome, max_price):.4f}",
            "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
            "post_only": False, "reduce_only": False, "cancel_order_on_pause": True,
        }
        d = self._call("POST", "/portfolio/events/orders", json=body)

        def f(k):
            return float(d[k]) if d.get(k) not in (None, "") else None

        return OrderAck(d["order_id"], d.get("client_order_id", cid), f("fill_count") or 0.0,
                        f("remaining_count") or 0.0, f("average_fill_price"), f("average_fee_paid"))
