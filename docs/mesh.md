# Compute mesh

With **Share compute** on (API drawer), a tab joins a pool. The ladder gains a tier: when your device cannot finish a conversion, another user's tab can, and yours can do the same for them. Text only; images never leave your device unless you opt in separately. Everything a peer returns goes through the same validator as every other tier.

## Rules the relay enforces

Nothing is self-reported. Busyness is what the relay dispatched and has not gotten back; liveness is the tab's own poll.

- **Good faith.** You can use peers only while your own tab is listening and pooled, and your weighted served/used ratio stays reasonable (equation 1, refine 2, prose 3, image 2, with a small grace for newcomers). Your own requests always go ahead of pool jobs on your tab.
- **Cheapest sufficient peer.** Jobs are classed (equation, prose, refine, check, image) and routed to the least-loaded peer capable of the class by expected finish time: queue ahead plus tokens divided by its measured tokens per second. Two candidates are sampled and the better one wins, so bursts do not stampede one peer. One pool job per tab at a time; a vanished peer's job is re-dispatched. Big models get a small penalty for trivial jobs so one 9B tab does not absorb all the traffic.
- **Spot checks.** One in ten pool jobs is duplicated to a second peer. Agreement feeds a per-peer reputation; low-agreement or failing peers stop receiving work.
- **Pools.** A blank passphrase joins the public pool; a passphrase creates a private pool (your laptop serving your phone, or a family).

Peer jobs run on the peer's headless pipeline with escalation disabled (they never re-escalate or use the peer's server) and are not written to the peer's history. The mesh tier sits after a self-hosted local server (trusted, fast) and before a cloud API (costs, logs).

## Why it is built this way

The tab already relays through the server for the Tab API, so a scheduler across tabs is a small addition to the relay. The hard problems are trust and incentive, not plumbing: the good-faith rule makes the mesh sustainable without accounts or credits, the validator and spot checks make results trustworthy, and pools give users a trust boundary they control.
