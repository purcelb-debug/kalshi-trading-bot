"""Authenticated Kalshi trading client: balance, positions, IOC limit orders, and resting
post-only buys (list and cancel).

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
    ("GET", "/portfolio/orders"),
}
CANCEL_PREFIX = "/portfolio/events/orders/"      # DELETE .../{order_id}, nothing deeper


def _allowed(method: str, path: str) -> bool:
    if (method, path) in ALLOWED:
        return True
    oid = path[len(CANCEL_PREFIX):] if path.startswith(CANCEL_PREFIX) else ""
    return method == "DELETE" and bool(oid) and "/" not in oid


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
        if not _allowed(method, path):
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

    def cash_by_exchange(self) -> dict[int, float]:
        """Available cash per Kalshi exchange (``balance_breakdown``). A market can only use
        the cash on its own exchange (``exchange_index`` on the market), e.g. weather on 0,
        crypto and commodities on 2 (seen 2026-11-04)."""
        d = self._call("GET", "/portfolio/balance")
        return {int(b["exchange_index"]): float(b["balance"])
                for b in d.get("balance_breakdown") or []}

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
        return self._ack(self._call("POST", "/portfolio/events/orders", json=body), cid)

    def place_ioc_exit(self, ticker: str, held: str, count: int, min_price: float,
                       client_order_id: str | None = None) -> OrderAck:
        """Immediate-or-cancel close of a held ``held`` ("yes"/"no") position: sells at
        >= ``min_price`` for that side, or not at all. ``reduce_only`` makes the exchange
        refuse anything that would open or grow a position."""
        if count < 1 or not 0.01 <= min_price <= 0.99:
            raise ValueError("count must be >= 1 and price within [0.01, 0.99]")
        if held not in ("yes", "no"):
            raise ValueError(held)
        cid = client_order_id or str(uuid.uuid4())
        # Selling YES = the YES ask side at >= p. Selling NO = buying YES at <= 1 - p.
        body = {
            "ticker": ticker, "client_order_id": cid,
            "side": "ask" if held == "yes" else "bid", "count": str(int(count)),
            "price": f"{(min_price if held == 'yes' else round(1 - min_price, 4)):.4f}",
            "time_in_force": "immediate_or_cancel", "self_trade_prevention_type": "taker_at_cross",
            "post_only": False, "reduce_only": True, "cancel_order_on_pause": True,
        }
        return self._ack(self._call("POST", "/portfolio/events/orders", json=body), cid)

    def place_maker_buy(self, ticker: str, outcome: str, count: int, price: float,
                        expires_ts: int, client_order_id: str | None = None) -> OrderAck:
        """Post-only resting limit buy at exactly ``price`` for ``outcome``: Kalshi rejects it
        rather than let it take liquidity, and cancels it at ``expires_ts`` (Unix seconds)."""
        if count < 1 or not 0.01 <= price <= 0.99:
            raise ValueError("count must be >= 1 and price within [0.01, 0.99]")
        cid = client_order_id or str(uuid.uuid4())
        body = {
            "ticker": ticker, "client_order_id": cid, "side": yes_book_side(outcome),
            "count": str(int(count)), "price": f"{yes_leg_price(outcome, price):.4f}",
            "time_in_force": "good_till_canceled", "expiration_time": int(expires_ts),
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": True, "reduce_only": False, "cancel_order_on_pause": True,
        }
        return self._ack(self._call("POST", "/portfolio/events/orders", json=body), cid)

    def orders(self, min_ts: int | None = None, status: str | None = None) -> list[dict]:
        """This account's orders: order_id, ticker, status, remaining, filled, and ``cost``,
        the price per contract of the side it buys (YES-leg bid = YES, ask = NO)."""
        q = {k: v for k, v in (("min_ts", min_ts), ("status", status)) if v is not None}
        out = []
        for o in self._pages("/portfolio/orders", "orders", q):
            yes_px = float(o.get("yes_price_dollars") or 0)
            buys_yes = o.get("book_side") == "bid" if o.get("book_side") else \
                (o.get("side") == "yes") == (o.get("action") == "buy")
            ex = o.get("exchange_index")
            out.append({"order_id": o["order_id"], "ticker": o.get("ticker"),
                        "status": o.get("status"), "exchange": None if ex is None else int(ex),
                        "remaining": float(o.get("remaining_count_fp") or 0),
                        "filled": float(o.get("fill_count_fp") or 0),
                        "cost": yes_px if buys_yes else round(1 - yes_px, 4)})
        return out

    def cancel_order(self, order_id: str, ticker: str) -> float:
        """Cancel a resting order; returns the contracts cancelled (its remaining count)."""
        d = self._call("DELETE", CANCEL_PREFIX + order_id, params={"market_ticker": ticker})
        return float(d.get("reduced_by") or 0)

    @staticmethod
    def _ack(d: dict, cid: str) -> OrderAck:
        def f(k):
            return float(d[k]) if d.get(k) not in (None, "") else None

        return OrderAck(d["order_id"], d.get("client_order_id", cid), f("fill_count") or 0.0,
                        f("remaining_count") or 0.0, f("average_fill_price"), f("average_fee_paid"))
