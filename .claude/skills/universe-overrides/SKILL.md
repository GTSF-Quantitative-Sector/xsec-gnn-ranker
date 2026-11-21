---
name: universe-overrides
description: Research the problems `gnns build-universe` reports (a failed build, or rows in data/sp500/reports/universe/) and propose fixes as rows in config/universes/overrides.csv, each with evidence, for the user to review. Use when a universe build fails on identifiers or Massive symbols, or when asked to review the universe reports.
---

# Universe overrides

Fixes to the universe are data, never code: rows in `config/universes/overrides.csv`
(`security_id, id_type, value, valid_from, valid_to, note`). A row replaces what the build
derived for that security and identifier type on the dates it covers (an empty `value`
only removes). For `ticker_massive`, the build also fetches that symbol's bars. Rules:
docs/CONTRACTS.md, "Identifiers". Never edit anything under `data/`: the next build would
overwrite it, and nobody could review it.

## Data rules (WRDS AI policy)

- **Allowed to read:** Datastream rows (names, tickers, CUSIPs, lists), Massive bars, the
  universe table's non-CRSP columns, and the build's error messages.
- **CRSP is counts only:** never print CRSP values or rows, even derived ones. That covers
  `permno`, `permco`, `ticker_crsp`, `mktcap`, and the reports
  `sessions_without_permno`, `member_sessions_without_mktcap`,
  `stopped_without_crsp_delisting`, `delistings_without_return` and everything in
  `reports/prices/`. Count their rows (e.g. `pl.read_csv(path).height`) and pass the
  details to the user.
- **Secrets:** never read `.env`.

## Steps

1. **Collect the problems.**
   - If the build failed, the message lists each security (by infocode) and the dates.
   - Otherwise, list `data/sp500/reports/universe/*.csv` with row counts, and read only
     the reports allowed above.
2. **Gather evidence for each security.**

       python scripts/override_evidence.py <security_id> --symbols <candidates> --start <d> --end <d>

   - **Candidates:** the Datastream tickers around the dates (the script prints them), the
     ticker the company used in that period, and class spellings (`BRKB` / `BRK.B`).
   - **What to look for:** a candidate is right when it has bars on the problem sessions
     and they continue the security's own bars at both ends (the script prints the log
     moves; within 0.2 counts as continuous). Check that the symbol didn't belong to
     another security on those dates: a jump at either end means it did.
3. **Choose the fix** using the table below. If nothing fits, or the evidence is
   ambiguous, don't write a row: report the security and what you found.
4. **Write the row.**
   - Make the dates the exact span the evidence supports: first and last session the
     symbol carries this security.
   - The note says what the source got wrong and what shows the fix is right, with dates
     and prices. Existing rows are the model, e.g. Sun Microsystems: "Datastream ticker
     history stops at SUNW (2007) and misses JAVA ...; Massive JAVA bars end 2010-01-26
     at 9.49 (deal price 9.50)".
5. **Rebuild and verify.**
   - Run `gnns build-universe`, then `gnns build-data`.
   - The problem must be gone, with no new failures and no new report rows for that
     security. The build's summary line "CRSP match on member sessions" must not get
     worse (an aggregate, fine to read).
6. **Hand over.** Commit the override rows on the working branch. Give the user one line
   per row (security, fix, evidence) plus anything left unfixed. The user reviews before
   any merge.

## Which fix for which problem

| Problem | Usual cause | Fix |
| :-- | :-- | :-- |
| Build fails: member sessions without a Massive symbol (more than 5 in a row), or `member_sessions_without_symbol` | Datastream's ticker history misses the ticker the security traded under | `ticker_massive` row for that ticker and span |
| `ticker_intervals_without_bars` | Massive spells the ticker differently, or the security wasn't trading yet | `ticker_massive` row with Massive's spelling; nothing if it wasn't trading |
| Build fails: two securities share an identifier on a date | a reused ticker, or Datastream switching a day early or late | a row with an empty `value` on the wrong security for those dates; or a `ticker_massive` row moving the span |
| `member_sessions_without_bar` | usually a trading halt | none, unless another symbol has bars that continue the price |
| `placeholder_list_entries` | Datastream placeholder lines | none (expected), unless the entry is a real security: then `ticker_datastream` / `cusip` rows from Datastream's own data |
| `invalid_bars_dropped` | bad vendor bars | none: the bars are dropped; report many on one security |
| `bars_after_crsp_delisting` | OTC trading after a delisting (expected) | none; report a security cut while it was clearly still listed |
| CRSP-based reports (above) | missing CRSP links or values | none by you: give the counts to the user. A missing PERMNO can come from a missing CUSIP: check Datastream's CUSIP history for a gap and propose a `cusip` row only if Datastream itself shows the value |
