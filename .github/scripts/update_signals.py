import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
JP_TICKERS = [f"16{i}.T" for i in range(17, 34)] # 1617 to 1633

US_TICKER_NAMES = {
    "XLB": "Materials", "XLC": "Communication", "XLE": "Energy", "XLF": "Financials", 
    "XLI": "Industrials", "XLK": "Technology", "XLP": "Consumer Staples", 
    "XLRE": "Real Estate", "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary"
}

JP_TICKER_NAMES = {
    "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材", "1620.T": "素材・化学",
    "1621.T": "医薬品", "1622.T": "自動車・輸送機", "1623.T": "鉄鋼・非鉄", "1624.T": "機械",
    "1625.T": "電機・精密", "1626.T": "情報通信・サービスその他", "1627.T": "電力・ガス", 
    "1628.T": "運輸・物流", "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
    "1632.T": "金融（除く銀行）", "1633.T": "不動産"
}

N_U = len(US_TICKERS)
N_J = len(JP_TICKERS)
N = N_U + N_J

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b)*b for b in basis)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis).T

def build_prior_subspace(c_full, us_tickers, jp_tickers):
    v1 = np.ones(N)
    v2 = np.zeros(N)
    v2[:N_U] = 1.0
    v2[N_U:] = -1.0
    v3 = np.zeros(N)
    
    us_cyc = ["XLB", "XLE", "XLF", "XLRE"]
    us_def = ["XLK", "XLP", "XLU", "XLV"]
    for i, tck in enumerate(us_tickers):
        if tck in us_cyc: v3[i] = 1.0
        elif tck in us_def: v3[i] = -1.0
        
    jp_cyc = ["1618.T", "1625.T", "1629.T", "1631.T"]
    jp_def = ["1617.T", "1621.T", "1627.T", "1630.T"]
    for i, tck in enumerate(jp_tickers):
        if tck in jp_cyc: v3[N_U + i] = 1.0
        elif tck in jp_def: v3[N_U + i] = -1.0
        
    V0 = gram_schmidt([v1, v2, v3])
    D0 = np.diag(np.diag(V0.T @ c_full @ V0))
    c_raw_0 = V0 @ D0 @ V0.T
    delta_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(c_raw_0)))
    c_0 = delta_inv_sqrt @ c_raw_0 @ delta_inv_sqrt
    np.fill_diagonal(c_0, 1.0)
    return c_0

def get_signal_for_t(rc_all, c_0, current_t, lreg=0.9, L=60):
    window = rc_all.iloc[current_t-L+1 : current_t+1]
    mean_u = window.iloc[:, :N_U].mean().values
    std_u = window.iloc[:, :N_U].std().values
    
    obs_u = rc_all.iloc[current_t, :N_U].values
    zs_u = (obs_u - mean_u) / (std_u + 1e-8)
    
    Ct = window.corr().values
    Ct_reg = (1 - lreg) * Ct + lreg * c_0
    
    evals_s, evecs_s = np.linalg.eigh(Ct_reg)
    top_k_vecs_s = evecs_s[:, -3:]
    V_U_s = top_k_vecs_s[:N_U, :]
    V_J_s = top_k_vecs_s[N_U:, :]
    
    f_t_s = V_U_s.T @ zs_u
    sig_pca_sub = V_J_s @ f_t_s
    
    sig_series = pd.Series(sig_pca_sub, index=JP_TICKERS)
    
    q = 0.3
    n = len(sig_series.dropna())
    k = max(1, int(n * q))
    
    ranks = sig_series.rank(method='first', na_option='keep')
    long_idx = ranks.nlargest(k).index
    short_idx = ranks.nsmallest(k).index
    
    return long_idx, short_idx, sig_series

def main():
    print("Downloading YF data...")
    us_data = yf.download(US_TICKERS, start="2010-01-01", progress=False)
    jp_data = yf.download(JP_TICKERS, start="2010-01-01", progress=False)
    
    us_close = us_data['Close'].dropna(how='all')
    jp_close = jp_data['Close'].dropna(how='all')
    
    common_dates = sorted(us_close.index.intersection(jp_close.index))
    us_close = us_close.loc[common_dates].ffill()
    jp_close = jp_close.loc[common_dates].ffill()
    
    rc_u = us_close.pct_change().dropna()
    rc_j = jp_close.pct_change().dropna()
    
    valid_dates = rc_u.index
    rc_u = rc_u.loc[valid_dates]
    rc_j = rc_j.loc[valid_dates]
    
    rc_all = pd.concat([rc_u, rc_j], axis=1)
    
    train_end = "2020-12-31"
    c_full = rc_all.loc[:train_end].corr().values
    c_0 = build_prior_subspace(c_full, US_TICKERS, JP_TICKERS)
    
    L = 60
    lreg = 0.9
    
    # --- 1. Evaluate Historical Performance Since 2026-04-01 ---
    eval_start_date = "2026-04-01"
    valid_indices = np.where(rc_all.index >= eval_start_date)[0]
    
    capital_per_etf = 100000
    cumulative_profit = 0
    history = []
    
    for D_idx in valid_indices:
        # We need the signal computed up to D_idx - 1 to formulate the return on D_idx
        signal_t = D_idx - 1
        if signal_t < L - 1:
            continue
            
        D_date_str = rc_all.index[D_idx].strftime("%Y-%m-%d")
        
        long_idx, short_idx, sig_series = get_signal_for_t(rc_all, c_0, signal_t, lreg, L)
        returns_on_D = rc_all.iloc[D_idx]
        
        daily_profit = 0
        details = []
        
        for tck in long_idx:
            ret = float(returns_on_D[tck])
            profit = capital_per_etf * ret
            daily_profit += profit
            details.append({
                "ticker": tck, 
                "name": JP_TICKER_NAMES.get(tck, tck),
                "type": "long", 
                "return": ret, 
                "profit": float(profit)
            })
            
        for tck in short_idx:
            ret = float(returns_on_D[tck])
            profit = capital_per_etf * (-ret) # shorting profit
            daily_profit += profit
            details.append({
                "ticker": tck, 
                "name": JP_TICKER_NAMES.get(tck, tck),
                "type": "short", 
                "return": ret, 
                "profit": float(profit)
            })
            
        cumulative_profit += daily_profit
        
        # Sort details by profit descending for nice display
        details = sorted(details, key=lambda x: x["profit"], reverse=True)
        
        history.append({
            "date": D_date_str,
            "daily_profit": float(daily_profit),
            "cumulative_profit": float(cumulative_profit),
            "details": details
        })

    # --- 2. Calculate Next JP Trading Day Signals ---
    t = len(rc_all) - 1
    t_date = rc_all.index[t]
    
    long_idx_next, short_idx_next, sig_series_next = get_signal_for_t(rc_all, c_0, t, lreg, L)
    
    longs = []
    shorts = []
    
    for tck in long_idx_next:
        longs.append({
            "ticker": tck,
            "name": JP_TICKER_NAMES.get(tck, tck),
            "score": float(sig_series_next[tck])
        })
        
    for tck in short_idx_next:
        shorts.append({
            "ticker": tck,
            "name": JP_TICKER_NAMES.get(tck, tck),
            "score": float(sig_series_next[tck])
        })
        
    longs = sorted(longs, key=lambda x: x['score'], reverse=True)
    shorts = sorted(shorts, key=lambda x: x['score'])
    
    # Structure final JSON
    output_data = {
        "calculated_at": datetime.now().isoformat(),
        "latest_us_close_date": t_date.strftime("%Y-%m-%d"),
        "target_jp_trade_date": "Next JP Trading Day",
        "long_etfs": longs,
        "short_etfs": shorts,
        "history": history
    }
    
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(repo_root, 'signals.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Daily signals successfully exported to {output_path}")

if __name__ == "__main__":
    main()
