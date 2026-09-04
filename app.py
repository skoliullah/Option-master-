import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

# ==========================================
# 1. PAGE CONFIG & PRO TERMINAL STYLING
# ==========================================
st.set_page_config(
    page_title="Institutional Quant Option Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { padding: 1.2rem; background-color: #0B0E14; }
    .stMetric {
        background-color: #151922;
        padding: 14px;
        border-radius: 10px;
        border: 1px solid #232836;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .order-box {
        background-color: #121824;
        border: 2px solid #2563EB;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
    }
    .order-field {
        background-color: #1E2638;
        border-left: 4px solid #3B82F6;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 6px;
    }
    .green-badge {
        background-color: #065F46;
        color: #34D399;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .red-badge {
        background-color: #881337;
        color: #F87171;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE PERSISTENCE (SQLite)
# ==========================================
DB_NAME = "options_scanner_v8.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS option_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                atm_strike REAL,
                pcr REAL,
                max_pain REAL,
                call_trap REAL,
                put_trap REAL,
                signal TEXT,
                otm3_strike REAL,
                otm3_type TEXT,
                otm3_ltp REAL,
                total_call_chng_oi REAL,
                total_put_chng_oi REAL
            )
        ''')
        conn.commit()

init_db()

def save_snapshot(data):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO option_snapshots 
            (atm_strike, pcr, max_pain, call_trap, put_trap, signal, otm3_strike, otm3_type, otm3_ltp, total_call_chng_oi, total_put_chng_oi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['atm'], data['pcr'], data['max_pain'], data['call_trap'], data['put_trap'],
            data['signal'], data['otm3_strike'], data['otm3_type'], data['otm3_ltp'],
            data['call_chng'], data['put_chng']
        ))
        conn.commit()

def get_last_two_snapshots():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM option_snapshots ORDER BY id DESC LIMIT 2')
        return c.fetchall()

def reset_database():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('DELETE FROM option_snapshots')
        conn.commit()

# ==========================================
# 3. CSV PARSER & STRUCTURAL QUANT ENGINE
# ==========================================
def parse_nse_csv(uploaded_file):
    cols = [
        'DUMMY_0',
        'CALL_OI', 'CALL_CHNG_IN_OI', 'CALL_VOLUME', 'CALL_IV', 'CALL_LTP', 'CALL_CHNG', 
        'CALL_BID_QTY', 'CALL_BID', 'CALL_ASK', 'CALL_ASK_QTY', 
        'STRIKE', 
        'PUT_BID_QTY', 'PUT_BID', 'PUT_ASK', 'PUT_ASK_QTY', 
        'PUT_CHNG', 'PUT_LTP', 'PUT_IV', 'PUT_VOLUME', 'PUT_CHNG_IN_OI', 'PUT_OI'
    ]
    df = pd.read_csv(uploaded_file, skiprows=2, names=cols, usecols=range(22))
    df = df.drop(columns=['DUMMY_0'])

    def clean_num(val):
        if pd.isna(val) or str(val).strip() in ['-', '']:
            return 0.0
        val_str = str(val).replace(',', '').strip()
        try:
            return float(val_str)
        except:
            return 0.0

    for c in df.columns:
        df[c] = df[c].apply(clean_num)

    df = df[df['STRIKE'] > 0].sort_values('STRIKE').reset_index(drop=True)
    return df

def analyze_option_data(df):
    total_call_oi = df['CALL_OI'].sum()
    total_put_oi = df['PUT_OI'].sum()
    total_call_chng_oi = df['CALL_CHNG_IN_OI'].sum()
    total_put_chng_oi = df['PUT_CHNG_IN_OI'].sum()
    total_call_vol = df['CALL_VOLUME'].sum()
    total_put_vol = df['PUT_VOLUME'].sum()

    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    # Put-Call Parity Synthetic Spot & ATM Detection
    valid_df = df[(df['CALL_LTP'] > 0) & (df['PUT_LTP'] > 0)].copy()
    if not valid_df.empty:
        valid_df['SYNTHETIC_SPOT'] = valid_df['STRIKE'] + valid_df['CALL_LTP'] - valid_df['PUT_LTP']
        spot_price = valid_df['SYNTHETIC_SPOT'].median()
        df['SPOT_DIFF'] = (df['STRIKE'] - spot_price).abs()
        atm_idx = df.sort_values('SPOT_DIFF').index[0]
        atm_strike = df.loc[atm_idx, 'STRIKE']
    else:
        atm_idx = len(df) // 2
        atm_strike = df.loc[atm_idx, 'STRIKE']
        spot_price = atm_strike

    # Institutional Max Pain Level
    strikes = df['STRIKE'].values
    call_oi = df['CALL_OI'].values
    put_oi = df['PUT_OI'].values
    total_losses = [np.sum(np.maximum(0, k - strikes) * call_oi + np.maximum(0, strikes - k) * put_oi) for k in strikes]
    max_pain_strike = strikes[np.argmin(total_losses)]

    # Major Institutional OI Walls
    call_wall = df.loc[df['CALL_OI'].idxmax(), 'STRIKE']  # Heavy Call Resistance
    put_wall = df.loc[df['PUT_OI'].idxmax(), 'STRIKE']    # Heavy Put Support

    # Relative Momentum Scoring Logic
    bullish_score = (total_put_chng_oi * 0.6) + (total_put_vol * 0.4)
    bearish_score = (total_call_chng_oi * 0.6) + (total_call_vol * 0.4)

    # Dynamic Delta Approximation for OTM-3 (~0.32 Delta)
    otm3_delta = 0.32

    if bullish_score >= bearish_score:
        signal = "BULLISH_MOMENTUM"
        otm3_idx = min(atm_idx + 3, len(df) - 1)
        otm3_strike = df.loc[otm3_idx, 'STRIKE']
        otm3_type = "CE"
        otm3_ltp = df.loc[otm3_idx, 'CALL_LTP']
        if otm3_ltp <= 0:
            otm3_ltp = df.loc[atm_idx, 'CALL_LTP']
        
        # STRUCTURAL TARGETS & SL (CE BUYING)
        target_spot_1 = max(max_pain_strike, atm_strike + 50)
        target_spot_2 = max(call_wall, target_spot_1 + 50)
        support_spot_sl = min(put_wall, atm_strike - 50)

        # Delta Conversion to Option Premium
        delta_target_1_pts = max(15.0, (target_spot_1 - spot_price) * otm3_delta)
        delta_target_2_pts = max(35.0, (target_spot_2 - spot_price) * otm3_delta)
        delta_sl_pts = max(12.0, (spot_price - support_spot_sl) * otm3_delta)

        target_fast = round(otm3_ltp + delta_target_1_pts, 2)
        target_trend = round(otm3_ltp + delta_target_2_pts, 2)
        structural_sl = round(max(5.0, otm3_ltp - delta_sl_pts), 2)
        
        total_breakeven = otm3_strike + otm3_ltp

    else:
        signal = "BEARISH_MOMENTUM"
        otm3_idx = max(atm_idx - 3, 0)
        otm3_strike = df.loc[otm3_idx, 'STRIKE']
        otm3_type = "PE"
        otm3_ltp = df.loc[otm3_idx, 'PUT_LTP']
        if otm3_ltp <= 0:
            otm3_ltp = df.loc[atm_idx, 'PUT_LTP']

        # STRUCTURAL TARGETS & SL (PE BUYING)
        target_spot_1 = min(max_pain_strike, atm_strike - 50)
        target_spot_2 = min(put_wall, target_spot_1 - 50)
        resistance_spot_sl = max(call_wall, atm_strike + 50)

        # Delta Conversion to Option Premium
        delta_target_1_pts = max(15.0, (spot_price - target_spot_1) * otm3_delta)
        delta_target_2_pts = max(35.0, (spot_price - target_spot_2) * otm3_delta)
        delta_sl_pts = max(12.0, (resistance_spot_sl - spot_price) * otm3_delta)

        target_fast = round(otm3_ltp + delta_target_1_pts, 2)
        target_trend = round(otm3_ltp + delta_target_2_pts, 2)
        structural_sl = round(max(5.0, otm3_ltp - delta_sl_pts), 2)

        total_breakeven = otm3_strike - otm3_ltp

    # Risk to Reward Ratio Calculation
    risk_pts = max(1.0, otm3_ltp - structural_sl)
    reward_pts = target_fast - otm3_ltp
    rrr = round(reward_pts / risk_pts, 2)

    return {
        'spot_price': spot_price,
        'atm': atm_strike,
        'pcr': pcr,
        'max_pain': max_pain_strike,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'signal': signal,
        'otm3_strike': otm3_strike,
        'otm3_type': otm3_type,
        'otm3_ltp': otm3_ltp,
        'target_fast': target_fast,
        'target_trend': target_trend,
        'structural_sl': structural_sl,
        'rrr': rrr,
        'total_breakeven': total_breakeven,
        'call_chng': total_call_chng_oi,
        'put_chng': total_put_chng_oi,
        'call_trap': call_wall,
        'put_trap': put_wall
    }

# ==========================================
# 4. MAIN DASHBOARD TERMINAL
# ==========================================
st.title("⚡ QUANT INSTITUTIONAL OPTION TERMINAL")
st.caption("OI-Wall Structural Stop Loss • Delta Converted Target Model • Naked Buying System")

st.subheader("📁 Upload NSE Option Chain CSV File")
uploaded_file = st.file_uploader(
    "Select your downloaded NSE Option Chain CSV file:", 
    type=None, 
    help="Tap here to select the CSV file from your phone or PC storage."
)

st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("🗑️ Reset Database History"):
        reset_database()
        st.success("History cleared!")

if uploaded_file is not None:
    try:
        df = parse_nse_csv(uploaded_file)
        if df.empty or len(df) < 5:
            st.error("❌ Invalid CSV format or empty file.")
        else:
            res = analyze_option_data(df)
            save_snapshot(res)

            st.success("✅ Option Chain Analyzed via Structural Delta Model!")

            # SECTION 1: MARKET STRUCTURE & DIRECTION
            st.markdown("### 📊 Structural Market Snapshot")
            m1, m2 = st.columns(2)
            m1.metric("📍 Synthetic Spot Price", f"{res['spot_price']:,.2f}")
            m2.metric("💀 Max Pain Pivot", f"{res['max_pain']:,.0f}")

            m3, m4 = st.columns(2)
            m3.metric("🛡️ Put Support Wall (Put OI)", f"{res['put_wall']:,.0f}")
            m4.metric("🚧 Call Resistance Wall (Call OI)", f"{res['call_wall']:,.0f}")

            st.markdown("---")

            # SECTION 2: SIGNAL & RRR
            c1, c2 = st.columns(2)
            if res['signal'] == "BULLISH_MOMENTUM":
                c1.markdown("#### **Signal Direction:** <span class='green-badge'>🟢 CALL BUYING (CE)</span>", unsafe_allow_html=True)
            else:
                c1.markdown("#### **Signal Direction:** <span class='red-badge'>🔴 PUT BUYING (PE)</span>", unsafe_allow_html=True)
            
            c2.metric("⚖️ Structural Risk-Reward Ratio (RRR)", f"1 : {res['rrr']}")

            st.markdown("---")

            # SECTION 3: DELTA SHIFT MATRIX
            st.markdown("### 🔄 Delta Shift & Writer Panic Tracker")
            history = get_last_two_snapshots()

            if len(history) > 1:
                curr = history[0]
                prev = history[1]

                shift_atm = curr[2] - prev[2]
                shift_mp = curr[4] - prev[4]

                d1, d2 = st.columns(2)
                d1.metric("📍 ATM Shift", f"{curr[2]:,.0f}", delta=f"{shift_atm:+,.0f} pts")
                d2.metric("💀 Max Pain Shift", f"{curr[4]:,.0f}", delta=f"{shift_mp:+,.0f} pts")

                st.markdown("#### 🚨 Option Writer Panic Alerts")
                if curr[2] > prev[4] and curr[11] < 0:
                    st.error(f"🔥 CALL WRITER PANIC: Spot breached Max Pain ({prev[4]:,.0f}) with Call Unwinding!")
                elif curr[2] < prev[4] and curr[12] < 0:
                    st.error(f"🔥 PUT WRITER PANIC: Spot dropped below Max Pain ({prev[4]:,.0f}) with Put Unwinding!")
                else:
                    st.info("ℹ️ Market structure is steady. Directional trend active.")
            else:
                st.warning("ℹ️ First CSV uploaded. Delta Shift will display on next CSV upload.")

            st.markdown("---")

            # SECTION 4: EXACT PRICE LEVELS
            st.markdown("### 🎯 Structural Target & Hard SL Levels")
            ltp = res['otm3_ltp']
            
            buy_limit_min = round(ltp * 0.95, 2)
            buy_limit_max = round(ltp * 0.99, 2)
            trailing_step = round(ltp * 0.05, 2)

            t1, t2 = st.columns(2)
            t1.metric("📌 Selected Strike", f"{res['otm3_strike']:,.0f} {res['otm3_type']}")
            t2.metric("🛒 Ideal Buying Limit Range", f"₹{buy_limit_min:.2f} - ₹{buy_limit_max:.2f}")

            t3, t4, t5 = st.columns(3)
            t3.metric("🎯 High-Probability Target 1", f"₹{res['target_fast']:.2f}")
            t4.metric("🚀 Extended Trend Target 2", f"₹{res['target_trend']:.2f}")
            t5.metric("🛡️ OI-Wall Structural SL", f"₹{res['structural_sl']:.2f}")

            st.markdown("---")

            # SECTION 5: ONE-CLICK BRACKET ORDER BLUEPRINT
            st.markdown("### 🚀 ONE-CLICK BRACKET ORDER BLUEPRINT")
            st.caption("Copy these exact values directly into your Zerodha / AngelOne / Groww / Dhan Bracket Order Form:")

            st.markdown(f"""
            <div class="order-box">
                <h3 style="color: #60A5FA; margin-top:0;">📋 Pro Order Blueprint ({res['otm3_strike']:,.0f} {res['otm3_type']})</h3>
                <div class="order-field">
                    <strong>1. Instrument:</strong> NIFTY {res['otm3_strike']:,.0f} {res['otm3_type']}
                </div>
                <div class="order-field">
                    <strong>2. Order Type:</strong> LIMIT / BRACKET ORDER (OCO / GTT)
                </div>
                <div class="order-field">
                    <strong>3. Buy Limit Price:</strong> ₹{buy_limit_max:.2f}
                </div>
                <div class="order-field">
                    <strong>4. Target 1 (Exit 70% Qty):</strong> ₹{res['target_fast']:.2f}
                </div>
                <div class="order-field">
                    <strong>5. Target 2 (Trail Remaining):</strong> ₹{res['target_trend']:.2f}
                </div>
                <div class="order-field">
                    <strong>6. Structural Hard Stop Loss:</strong> ₹{res['structural_sl']:.2f} <i>(Safe from market noise/spikes)</i>
                </div>
                <div class="order-field">
                    <strong>7. Total Price Break-Even Level:</strong> ₹{res['total_breakeven']:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📋 Quick Copy Text for Trading Terminal:")
            copy_text = f"SYMBOL: NIFTY {res['otm3_strike']:,.0f} {res['otm3_type']} | BUY_LIMIT: {buy_limit_max:.2f} | TARGET_1: {res['target_fast']:.2f} | TARGET_2: {res['target_trend']:.2f} | STRUCTURAL_SL: {res['structural_sl']:.2f}"
            st.code(copy_text, language="text")

    except Exception as e:
        st.error(f"❌ Error processing CSV file: {e}")
