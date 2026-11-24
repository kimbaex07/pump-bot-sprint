// pump-bot-sprint: momentum bot for pump.fun-style bonding curves.
//
// Idea, in one paragraph: only chase a curve that already looks alive —
// mature (real SOL raised and dozens of trades), with fresh buy flow that is
// accelerating (buy volume far above sell volume, spread over many distinct
// buyers), with the price climbing and still sitting near its recent high.
// One small fixed-size buy per coin, never add to a position, and get out fast
// on any sign the move is over. The replay is scored on paper net PnL over
// coins the house records after entries close, so the bot must be robust on
// unseen coins, not tuned to one tape: every threshold below is a round,
// deliberately conservative number, justified where it is used.
//
// Determinism: the bot only reads the batch stream. No clock, no environment,
// no Math.random. Every map is iterated in insertion order, so two runs of
// the same replay produce the same orders byte for byte.

import { runBot, LAMPORTS_PER_SOL, type Batch, type BotState, type OrderInput } from './vendor/makefun-bot.ts';

// ---------------------------------------------------------------- tuning
// All sizes are lamports (1 SOL = 1_000_000_000 lamports).

// Fixed buy per position. 0.15 SOL keeps a single loser small (about 1.5% of
// the 10 SOL bankroll) while staying well above the 0.01 SOL round-trip floor
// the guardrail counts, so a filled buy plus its exit always counts.
const BUY_SIZE_LAMPORTS = 150_000_000n;
// Never enter with less than this actually spendable: a buy capped below
// 0.05 SOL is mostly eaten by the per-order tx fee (105_000 lamports) and
// the curve fees, so it is not worth the slot.
const MIN_BUY_LAMPORTS = 50_000_000n;

// Maturity: the curve must have raised real SOL and survived a decent number
// of trades before we touch it. A brand-new curve is where rugs and one-trade
// wonders live; 25 SOL raised + 45 trades filters most of that out while
// still leaving room to catch a run.
const MIN_TRADES = 50;
const MIN_REAL_SOL_LAMPORTS = 25_000_000_000n;

// Lookback window for flow and price: the last 20 trades on the curve.
const WINDOW_TRADES = 20;
// Buy flow must dominate: buy quote volume at least 4x sell quote volume
// across the window. A 4:1 ratio means the tape is decisively net-long.
const MIN_BUY_SELL_RATIO_NUM = 4n;
const MIN_BUY_SELL_RATIO_DEN = 1n;
// ...and the buying must come from at least 10 distinct wallets, so one
// whale washing a curve cannot trigger us.
const MIN_DISTINCT_BUYERS = 10;
// Price must be up at least 35% from the start of the window...
const MIN_WINDOW_GAIN_BPS = 3_500n;
// ...and still within 5% of the window high, so we buy strength, not a coin
// that already rolled over and is sliding back down.
const MAX_OFF_HIGH_BPS = 500n;
// Acceleration: buy volume in the second half of the window must be at
// least 1.5x buy volume in the first half. Steady buying is not momentum;
// buying that is picking up speed is.
const ACCEL_NUM = 3n;
const ACCEL_DEN = 2n;

// Exits (all measured against the filled entry price per token).
// Stop-loss: down 10% from entry. Cut losers fast: a momentum buy that is
// wrong is usually wrong quickly, and small losses are what let the rare
// big winners pay for the many small losers.
const STOP_LOSS_BPS = 1_000n;
// Take-profit: up 60% from entry. On this tape most runs top out well
// before a multi-x move, so bank a solid gain when it arrives instead of
// watching it retrace back to the trailing stop.
const TAKE_PROFIT_BPS = 6_000n;
// Trailing stop: down 12% from the best price seen since entry. Tight
// enough to keep most of a 60%+ run once it turns, wide enough to survive
// a pump coin's normal pullbacks.
const TRAILING_BPS = 1_200n;
// Fade exit: price 10% off its peak AND this batch's flow flipped to net
// selling. A fade with sellers in control is a top, not a dip.
const FADE_OFF_PEAK_BPS = 1_000n;
// Big-sell exit: one batch's sell volume is 3x its buy volume and at least
// 0.5 SOL — somebody large is leaving; leave with them.
const BIG_SELL_RATIO = 3n;
const BIG_SELL_MIN_LAMPORTS = 500_000_000n;
// Max hold: 9_000 slots (~1 hour at ~0.4 s/slot). A momentum buy that has
// not paid off within an hour is a stale position tying up a slot; free it
// so the bot can cycle to fresh coins (the guardrail wants 20+ distinct
// mints round-tripped).
const MAX_HOLD_SLOTS = 9_000;

// Slippage allowance on every order: 5%.
const SLIPPAGE_BPS = 500;

// Scale factor for integer price math: price = quote_reserves * SCALE /
// token_reserves, i.e. lamports per token scaled by 1e9. All comparisons use
// cross-multiplied bigints, so no floating point ever appears.
const SCALE = 1_000_000_000n;

// ---------------------------------------------------------------- per-mint tape

interface Tick {
  slot: number;
  buy: boolean;
  quote: bigint; // SOL moved by this trade, in lamports
  price: bigint; // scaled price just after this trade
  user: string;
}

interface MintTape {
  // Last WINDOW_TRADES trades, oldest first.
  window: Tick[];
}

interface Held {
  entryPrice: bigint; // scaled price actually paid (from the buy fill)
  peakPrice: bigint; // best scaled price since entry
  sinceSlot: number; // slot the buy fill landed
}

const tapes = new Map<string, MintTape>();
const held = new Map<string, Held>();

function tapeFor(mint: string): MintTape {
  let t = tapes.get(mint);
  if (t === undefined) {
    t = { window: [] };
    tapes.set(mint, t);
  }
  return t;
}

/** Scaled price of a curve from its virtual reserves (lamports per token). */
function priceOf(state: BotState, mint: string): bigint | null {
  const c = state.curves.get(mint);
  if (c === undefined) return null;
  if (c.virtual_token_reserves <= 0n || c.virtual_quote_reserves <= 0n) return null;
  return (c.virtual_quote_reserves * SCALE) / c.virtual_token_reserves;
}

/** True when price `p` is at least `bps` basis points above `base`. */
function upAtLeast(p: bigint, base: bigint, bps: bigint): boolean {
  return p * 10_000n >= base * (10_000n + bps);
}

/** True when price `p` is at least `bps` basis points below `base`. */
function downAtLeast(p: bigint, base: bigint, bps: bigint): boolean {
  return p * 10_000n <= base * (10_000n - bps);
}

// ---------------------------------------------------------------- the bot

await runBot({
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const orders: OrderInput[] = [];

    // -- 1. Fold this batch's fills into our held-position bookkeeping. --
    // A buy fill fixes the true entry price (SOL actually spent per token).
    // A sell fill, or the harness closing a completed/flattened position,
    // removes the coin from the book.
    for (const fill of batch.fills) {
      if (fill.status !== 'filled') continue;
      if (fill.side === 'buy') {
        const tokens = BigInt(fill.token_amount);
        const spent = BigInt(fill.quote_lamports);
        if (tokens > 0n && spent > 0n) {
          const entry = (spent * SCALE) / tokens;
          held.set(fill.mint, { entryPrice: entry, peakPrice: entry, sinceSlot: fill.slot });
        }
      } else {
        held.delete(fill.mint);
      }
    }

    // -- 2. Fold this batch's events into the per-mint tapes, and collect
    //    this batch's per-mint buy/sell flow for the exit checks. --
    const batchFlow = new Map<string, { buy: bigint; sell: bigint }>();
    const tradedMints: string[] = [];
    const seenThisBatch = new Set<string>();
    for (const e of batch.events) {
      if (e.event !== 'TradeEvent') {
        if (e.event === 'CompleteEvent') {
          // Curve over: drop its tape; any position is closed by the harness.
          tapes.delete(e.mint);
          held.delete(e.mint);
        }
        continue;
      }
      const flow = batchFlow.get(e.mint) ?? { buy: 0n, sell: 0n };
      const quote = BigInt(e.sol_amount);
      if (e.is_buy) flow.buy += quote;
      else flow.sell += quote;
      batchFlow.set(e.mint, flow);

      const tape = tapeFor(e.mint);
      const price =
        BigInt(e.virtual_quote_reserves) > 0n && BigInt(e.virtual_token_reserves) > 0n
          ? (BigInt(e.virtual_quote_reserves) * SCALE) / BigInt(e.virtual_token_reserves)
          : 0n;
      tape.window.push({ slot: e.slot, buy: e.is_buy, quote, price, user: e.user });
      if (tape.window.length > WINDOW_TRADES) tape.window.shift();
      if (!seenThisBatch.has(e.mint)) {
        seenThisBatch.add(e.mint);
        tradedMints.push(e.mint);
      }
    }

    // Drop held records the harness already flattened (positions the account
    // no longer reports and with no sell still pending).
    const pendingSell = new Set<string>();
    for (const p of state.pending.values()) {
      if (p.order.side === 'sell') pendingSell.add(p.order.mint);
    }
    for (const mint of [...held.keys()]) {
      if (!state.positions.has(mint) && !pendingSell.has(mint)) held.delete(mint);
    }

    // -- 3. EXITS FIRST: every held coin is checked before any new buy. --
    for (const [mint, h] of held) {
      const pos = state.positions.get(mint);
      if (pos === undefined || pos.token_amount <= 0n) continue;
      if (pendingSell.has(mint)) continue; // an exit is already in flight
      const price = priceOf(state, mint);
      if (price === null || price <= 0n) continue;
      if (price > h.peakPrice) h.peakPrice = price;

      const flow = batchFlow.get(mint);
      const batchSellDominated =
        flow !== undefined &&
        flow.sell >= BIG_SELL_MIN_LAMPORTS &&
        flow.sell >= flow.buy * BIG_SELL_RATIO;
      const faded =
        downAtLeast(price, h.peakPrice, FADE_OFF_PEAK_BPS) &&
        flow !== undefined &&
        flow.sell > flow.buy;

      let exit = false;
      if (downAtLeast(price, h.entryPrice, STOP_LOSS_BPS)) exit = true; // stop-loss
      else if (upAtLeast(price, h.entryPrice, TAKE_PROFIT_BPS)) exit = true; // take-profit
      else if (downAtLeast(price, h.peakPrice, TRAILING_BPS)) exit = true; // trailing stop
      else if (faded) exit = true; // momentum faded with sellers in control
      else if (batchSellDominated) exit = true; // big seller leaving
      else if (batch.slot - h.sinceSlot >= MAX_HOLD_SLOTS) exit = true; // stale position

      if (exit) {
        orders.push({
          side: 'sell',
          mint,
          token_amount: pos.token_amount,
          max_slippage_bps: SLIPPAGE_BPS,
        });
      }
    }

    // -- 4. ENTRIES: only coins that traded in this batch, one fixed buy. --
    if (state.committedPositionCount() < state.hello.caps.max_open_positions) {
      for (const mint of tradedMints) {
        if (orders.length >= state.hello.caps.max_orders_per_batch) break;
        if (state.committedPositionCount() + countBuysSoFar(orders) >= state.hello.caps.max_open_positions) break;
        if (state.positions.has(mint) || held.has(mint)) continue; // already in, no pyramiding
        if (hasPendingBuy(state, mint)) continue;

        const curve = state.curves.get(mint);
        if (curve === undefined || curve.complete) continue;
        // Maturity: enough trades and real SOL raised.
        if (curve.trades < MIN_TRADES) continue;
        if (curve.real_quote_reserves < MIN_REAL_SOL_LAMPORTS) continue;

        const tape = tapes.get(mint);
        if (tape === undefined || tape.window.length < WINDOW_TRADES) continue;
        const price = priceOf(state, mint);
        if (price === null || price <= 0n) continue;

        // Window flow: buy volume, sell volume, distinct buyers, and whether
        // the buying is accelerating (second-half buy volume vs first half).
        let buyVol = 0n;
        let sellVol = 0n;
        let buyFirstHalf = 0n;
        let buySecondHalf = 0n;
        const buyers = new Set<string>();
        let windowHigh = 0n;
        const half = Math.floor(tape.window.length / 2);
        for (let i = 0; i < tape.window.length; i++) {
          const t = tape.window[i];
          if (t.buy) {
            buyVol += t.quote;
            buyers.add(t.user);
            if (i < half) buyFirstHalf += t.quote;
            else buySecondHalf += t.quote;
          } else {
            sellVol += t.quote;
          }
          if (t.price > windowHigh) windowHigh = t.price;
        }
        // Accelerating net buy flow from many distinct buyers.
        if (buyVol * MIN_BUY_SELL_RATIO_DEN < sellVol * MIN_BUY_SELL_RATIO_NUM) continue;
        if (buyers.size < MIN_DISTINCT_BUYERS) continue;
        if (buySecondHalf * ACCEL_DEN < buyFirstHalf * ACCEL_NUM) continue;
        // Price up over the window and near its window high.
        const windowStart = tape.window[0].price;
        if (windowStart <= 0n) continue;
        if (!upAtLeast(price, windowStart, MIN_WINDOW_GAIN_BPS)) continue;
        if (windowHigh > 0n && downAtLeast(price, windowHigh, MAX_OFF_HIGH_BPS)) continue;

        // Flat sizing, capped by the per-order reserve cap and by free cash.
        let size = BUY_SIZE_LAMPORTS;
        const cap = state.maxBuyLamports(mint);
        if (cap < size) size = cap;
        const free = state.freeCashLamports() - spentByOrdersSoFar(orders, state);
        if (size > free) size = free;
        if (size < MIN_BUY_LAMPORTS) continue;

        orders.push({
          side: 'buy',
          mint,
          quote_lamports: size,
          max_slippage_bps: SLIPPAGE_BPS,
        });
      }
    }

    return orders;
  },

  onDone(state: BotState): void {
    console.error(
      `done: batches=${state.seq + 1} cash=${state.cash} positions=${state.positions.size} curves_seen=${state.curves.size}`,
    );
  },
});

function hasPendingBuy(state: BotState, mint: string): boolean {
  for (const p of state.pending.values()) {
    if (p.order.side === 'buy' && p.order.mint === mint) return true;
  }
  return false;
}

function countBuysSoFar(orders: OrderInput[]): number {
  let n = 0;
  for (const o of orders) if (o.side === 'buy') n++;
  return n;
}

function spentByOrdersSoFar(orders: OrderInput[], state: BotState): bigint {
  // Orders already queued this batch will each pay a tx fee if they fill;
  // reserve that plus the buy amounts so later buys in the same batch do
  // not over-commit the cash.
  let spent = 0n;
  const tx = state.txFeeLamports();
  for (const o of orders) {
    spent += tx;
    if (o.side === 'buy') spent += BigInt(o.quote_lamports);
  }
  return spent;
}
