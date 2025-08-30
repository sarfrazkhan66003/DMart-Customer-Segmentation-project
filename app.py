# app.py
# ---------------------------------------------------------
# DMart-like Customer Segmentation (Streamlit + SQLite + ML)
# Upgraded: Forecasting, Recommendations, Churn Prediction,
# Real-time filters + drilldowns, Heatmap Categories vs Segments
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import joblib
import plotly.express as px

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="DMart Customer Segmentation — Enhanced", page_icon="🛒", layout="wide")
st.title("🛒 DMart-style Customer Segmentation — Enhanced")

DB_PATH = "dmart.db"
MODEL_DIR = "models"
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CHURN_PATH = os.path.join(MODEL_DIR, "churn.pkl")
CHURN_SCALER_PATH = os.path.join(MODEL_DIR, "churn_scaler.pkl")

# -----------------------------
# Utilities: DB
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    os.makedirs(MODEL_DIR, exist_ok=True)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INTEGER,
            city TEXT,
            loyalty_member INTEGER,
            signup_date TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id TEXT PRIMARY KEY,
            customer_id TEXT,
            tx_date TEXT,
            category TEXT,
            product TEXT,
            channel TEXT,
            quantity INTEGER,
            gross_amount REAL,
            discount REAL,
            net_amount REAL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        )""")
        conn.commit()

def table_counts():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM customers"); n_cust = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM transactions"); n_tx = c.fetchone()[0]
    return n_cust, n_tx

# -----------------------------
# Synthetic Data (DMart-like)
# -----------------------------
CITIES = ["Mumbai","Pune","Nashik","Nagpur","Thane","Surat","Ahmedabad","Bengaluru","Hyderabad"]
CATS   = ["Groceries","Fresh Produce","Dairy","Bakery","Home Care","Personal Care","Beverages","Snacks","Baby Care","Kitchen"]
CHANNELS = ["In-Store","Online"]
PRODUCTS_BY_CAT = {
    "Groceries": ["Atta","Rice 5kg","Sugar","Salt","Masala Mix"],
    "Fresh Produce": ["Tomato 1kg","Potato 1kg","Onion 1kg","Apple 1kg"],
    "Dairy": ["Milk 1L","Paneer 200g","Butter 100g","Curd 1kg"],
    "Bakery": ["Bread","Buns","Cake Slice","Cookies"],
    "Home Care": ["Detergent 1kg","Floor Cleaner","Dishwash Liquid"],
    "Personal Care": ["Shampoo 200ml","Soap","Toothpaste"],
    "Beverages": ["Cola 1L","Juice 1L","Tea 500g"],
    "Snacks": ["Chips 100g","Namkeen 200g","Chocolate"],
    "Baby Care": ["Diaper Pack","Baby Food"],
    "Kitchen": ["Pan","Knife","Spatula"]
}

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)

def random_name():
    first = ["Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Ishaan","Krishna","Ayaan","Atharv",
             "Anaya","Siya","Myra","Aadhya","Anika","Sara","Mira","Zara","Riya","Aarohi"]
    last = ["Sharma","Patel","Reddy","Iyer","Khan","Gupta","Desai","Kulkarni","Shetty","Naidu"]
    return f"{random.choice(first)} {random.choice(last)}"

def cat_price(cat):
    base = {
        "Groceries": (200, 50),
        "Fresh Produce": (120, 40),
        "Dairy": (160, 40),
        "Bakery": (150, 60),
        "Home Care": (300, 120),
        "Personal Care": (220, 90),
        "Beverages": (180, 70),
        "Snacks": (140, 60),
        "Baby Care": (450, 160),
        "Kitchen": (350, 140),
    }
    mu, sigma = base[cat]
    return max(30, np.random.normal(mu, sigma))

def generate_customers(n=1200):
    rows = []
    today = datetime.today()
    for i in range(n):
        cid = f"C{100000+i}"
        gender = random.choice(["M","F"])
        age = int(np.clip(np.random.normal(34, 10), 18, 70))
        city = random.choice(CITIES)
        loyalty = 1 if random.random() < 0.55 else 0
        signup = (today - timedelta(days=random.randint(30, 1200))).date().isoformat()
        rows.append((cid, random_name(), gender, age, city, loyalty, signup))
    return rows

def generate_transactions(customers, n_tx=22000):
    rows = []
    today = datetime.today().date()
    for (cid, *_rest) in customers:
        freq_factor = np.clip(np.random.normal(1.0, 0.35), 0.4, 2.2)
        tx_count = int(np.clip(np.random.poisson(8 * freq_factor), 1, 80))
        for _ in range(tx_count):
            tx_date = today - timedelta(days=random.randint(0, 365))
            items = np.random.choice(range(1, 9), p=[0.18,0.22,0.20,0.15,0.10,0.08,0.05,0.02])
            for _i in range(items):
                cat = random.choices(CATS, weights=[20,10,12,8,10,8,7,12,5,8], k=1)[0]
                prod = random.choice(PRODUCTS_BY_CAT.get(cat, [f"{cat} Item"]))
                channel = random.choices(CHANNELS, weights=[75,25], k=1)[0]
                qty = int(np.clip(np.random.poisson(2), 1, 10))
                price = cat_price(cat) * qty
                disc = price * random.choice([0.0,0.05,0.10,0.15]) if random.random()<0.6 else 0.0
                net = price - disc
                txid = f"T{cid}_{tx_date}_{random.randint(1000,9999)}_{_i}"
                rows.append((txid, cid, tx_date.isoformat(), cat, prod, channel, qty, round(price,2), round(disc,2), round(net,2)))
    if len(rows) > n_tx:
        rows = random.sample(rows, n_tx)
    return rows

def populate_if_empty():
    n_c, n_t = table_counts()
    if n_c == 0 or n_t == 0:
        seed_everything(42)
        customers = generate_customers(1200)
        txs = generate_transactions(customers, n_tx=22000)
        with get_conn() as conn:
            c = conn.cursor()
            c.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?,?)", customers)
            c.executemany("INSERT OR REPLACE INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?)", txs)
            conn.commit()

# -----------------------------
# Feature Engineering (RFM etc.)
# -----------------------------
def compute_features(as_of=None):
    if as_of is None:
        as_of = pd.to_datetime(datetime.today())

    with get_conn() as conn:
        tx = pd.read_sql_query("SELECT * FROM transactions", conn, parse_dates=["tx_date"])
        cust = pd.read_sql_query("SELECT * FROM customers", conn, parse_dates=["signup_date"])

    if tx.empty:
        return pd.DataFrame()

    tx['tx_date'] = pd.to_datetime(tx['tx_date'])
    last_tx = tx.groupby('customer_id')['tx_date'].max()
    r = (as_of - last_tx).dt.days.rename("recency_days")
    f = tx.groupby('customer_id')['tx_id'].nunique().rename("frequency")
    m = tx.groupby('customer_id')['net_amount'].sum().rename("monetary")
    avg_basket = tx.groupby('customer_id').apply(
        lambda d: d['net_amount'].sum() / max(1, d['tx_id'].nunique())
    ).rename("avg_basket_value")
    online_ratio = (
        tx.assign(is_online=(tx['channel'] == "Online").astype(int))
          .groupby('customer_id')['is_online'].mean()
          .rename("online_ratio")
    )

    cat_share = (
        tx.pivot_table(index='customer_id', columns='category',
                       values='net_amount', aggfunc='sum', fill_value=0)
    )
    # normalize shares
    total_cat = cat_share.sum(axis=1).replace(0, 1)
    cat_share = cat_share.div(total_cat, axis=0).add_prefix("share_")
    top_cols = sorted(cat_share.sum().sort_values(ascending=False).index[:4])
    cat_share = cat_share[top_cols]

    feats = pd.concat([r, f, m, avg_basket, online_ratio, cat_share], axis=1).reset_index()
    feats = feats.merge(cust[['customer_id','city','loyalty_member','age','gender']], on='customer_id', how='left')
    feats['loyalty_member'] = feats['loyalty_member'].fillna(0)
    return feats

# -----------------------------
# Modeling: Segments (KMeans)
# -----------------------------
def train_segments(df, k=4):
    use_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
               [c for c in df.columns if c.startswith('share_')]
    X = df[use_cols].fillna(0).copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    df2 = df.copy()
    df2['segment'] = labels
    # save
    joblib.dump(km, KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return df2, sil, use_cols

def load_segment_model():
    if os.path.exists(KMEANS_PATH) and os.path.exists(SCALER_PATH):
        km = joblib.load(KMEANS_PATH)
        scaler = joblib.load(SCALER_PATH)
        return km, scaler
    return None, None

def predict_segments(df, use_cols):
    km, scaler = load_segment_model()
    if km is None:
        raise RuntimeError("Segment model not trained yet.")
    X = df[use_cols].fillna(0)
    Xs = scaler.transform(X)
    out = df.copy()
    out['segment'] = km.predict(Xs)
    return out

def label_segments(df):
    summary = df.groupby('segment').agg(
        customers=('customer_id','nunique'),
        recency_days=('recency_days','mean'),
        frequency=('frequency','mean'),
        monetary=('monetary','mean'),
        avg_basket_value=('avg_basket_value','mean'),
        online_ratio=('online_ratio','mean')
    ).reset_index()
    summary['value_score'] = (
        summary['monetary'].rank(ascending=False) +
        summary['frequency'].rank(ascending=False) +
        summary['recency_days'].rank(ascending=True)
    )
    order = summary.sort_values('value_score').segment.tolist()
    names = ["Champions","Loyal Big Spenders","Potential Loyalists","Hibernating/At-Risk","New Customers","Occasional Buyers","Price Sensitive"]
    mapping = {seg: names[i] if i < len(names) else f"Segment {seg}" for i, seg in enumerate(order)}
    df['segment_name'] = df['segment'].map(mapping)
    summary['segment_name'] = summary['segment'].map(mapping)
    return df, summary.sort_values('value_score')

# -----------------------------
# Modeling: Churn Prediction
# -----------------------------
def create_churn_labels(feats, churn_days=90):
    # churn if recency_days > churn_days
    feats = feats.copy()
    feats['churn'] = (feats['recency_days'] > churn_days).astype(int)
    return feats

def train_churn_model(feats):
    # require feats containing recency, frequency, monetary etc
    df = create_churn_labels(feats)
    use_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
               [c for c in df.columns if c.startswith('share_')]
    X = df[use_cols].fillna(0).copy()
    y = df['churn']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    joblib.dump(clf, CHURN_PATH)
    joblib.dump(scaler, CHURN_SCALER_PATH)
    return clf, scaler, score, use_cols

def load_churn_model():
    if os.path.exists(CHURN_PATH) and os.path.exists(CHURN_SCALER_PATH):
        clf = joblib.load(CHURN_PATH)
        scaler = joblib.load(CHURN_SCALER_PATH)
        return clf, scaler
    return None, None

def predict_churn_for_row(row, use_cols):
    clf, scaler = load_churn_model()
    if clf is None:
        raise RuntimeError("Churn model not trained.")
    X = row[use_cols].fillna(0).values.reshape(1, -1)
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[0,1]
    pred = clf.predict(Xs)[0]
    return pred, proba

# -----------------------------
# Forecasting (per category) — simple linear trend on monthly totals
# -----------------------------
def forecast_category(category, months_ahead=3):
    with get_conn() as conn:
        tx = pd.read_sql_query("SELECT tx_date, category, net_amount FROM transactions WHERE category=?", conn, params=[category], parse_dates=["tx_date"])
    if tx.empty:
        return None
    tx['tx_date'] = pd.to_datetime(tx['tx_date'])
    tx['period'] = tx['tx_date'].dt.to_period('M').dt.to_timestamp()
    monthly = tx.groupby('period')['net_amount'].sum().reset_index().sort_values('period')
    if len(monthly) < 3:
        # not enough history -> return monthly with moving average forecast
        last = monthly['net_amount'].iloc[-3:].mean() if len(monthly) >= 1 else 0
        future = []
        last_period = monthly['period'].max() if not monthly.empty else pd.to_datetime(datetime.today()).to_period('M').to_timestamp()
        for i in range(1, months_ahead+1):
            future.append((last_period + pd.DateOffset(months=i), last))
        future_df = pd.DataFrame(future, columns=['period','forecast'])
        monthly = monthly.rename(columns={'net_amount':'actual'})
        return monthly, future_df

    # encode periods
    monthly = monthly.reset_index(drop=True)
    monthly['t'] = np.arange(len(monthly))
    X = monthly[['t']].values
    y = monthly['net_amount'].values
    lm = LinearRegression()
    lm.fit(X, y)
    last_t = monthly['t'].iloc[-1]
    future_t = np.arange(last_t+1, last_t+1+months_ahead).reshape(-1,1)
    preds = lm.predict(future_t)
    future_periods = [monthly['period'].iloc[-1] + pd.DateOffset(months=i) for i in range(1, months_ahead+1)]
    future_df = pd.DataFrame({'period': future_periods, 'forecast': np.maximum(preds, 0)})
    monthly = monthly.rename(columns={'net_amount':'actual'})
    return monthly[['period','actual']], future_df

# -----------------------------
# Product Recommendations (top sellers + trending)
# -----------------------------
def recommend_products(category, top_n=5):
    with get_conn() as conn:
        tx = pd.read_sql_query("SELECT tx_date, category, product, net_amount FROM transactions WHERE category=?", conn, params=[category], parse_dates=["tx_date"])
    if tx.empty:
        return []
    tx['tx_date'] = pd.to_datetime(tx['tx_date'])
    # total sales
    tot = tx.groupby('product')['net_amount'].sum().rename('total').reset_index()
    # recent sales (last 30 days)
    cutoff = pd.to_datetime(datetime.today()) - pd.Timedelta(days=30)
    recent = tx[tx['tx_date'] >= cutoff].groupby('product')['net_amount'].sum().rename('recent').reset_index()
    df = tot.merge(recent, on='product', how='left').fillna(0)
    df['trend_score'] = (df['recent'] + 1) / (df['total'] + 1)  # simple trend metric
    df = df.sort_values(['trend_score','total'], ascending=[False, False])
    return df.head(top_n).to_dict('records')

# -----------------------------
# UI: Sidebar / Init
# -----------------------------
init_db()
populate = st.sidebar.checkbox("🔁 (Re)generate synthetic DMart data", value=False,
                               help="Recreate demo data if you want a fresh run")
if populate:
    with get_conn() as conn:
        conn.execute("DELETE FROM transactions")
        conn.execute("DELETE FROM customers")
        conn.commit()
    populate_if_empty()
else:
    populate_if_empty()

n_c, n_t = table_counts()
st.sidebar.success(f"DB Ready: {n_c} customers, {n_t} transactions")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🧠 Train / Retrain Models",
    "🔎 Segment Explorer",
    "🔮 Forecast & Recommendations",
    "💡 Churn Prediction",
    "👤 Customer Lookup",
    "📤 Upload & Score CSV"
])

# -----------------------------
# Helper: Filters UI
# -----------------------------
def filters_ui(tx):
    st.sidebar.markdown("### Filters")
    cities = sorted(tx['city'].unique().tolist())
    cats = sorted(tx['category'].unique().tolist())
    segs = tx.get('segment_name', pd.Series()).unique().tolist() if 'segment_name' in tx else []
    city = st.sidebar.multiselect("City", options=cities, default=[])
    category = st.sidebar.multiselect("Category", options=cats, default=[])
    loyalty = st.sidebar.selectbox("Loyalty member", options=["All","Yes","No"], index=0)
    date_from = st.sidebar.date_input("Date from", value=(pd.to_datetime(datetime.today()) - pd.Timedelta(days=365)).date())
    date_to = st.sidebar.date_input("Date to", value=pd.to_datetime(datetime.today()).date())
    return dict(city=city, category=category, loyalty=loyalty, date_from=date_from, date_to=date_to)

# -----------------------------
# Page: Overview
# -----------------------------
if page == "📊 Overview":
    feats = compute_features()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{feats.customer_id.nunique():,}")
    col2.metric("Avg Monetary (₹)", f"{feats['monetary'].mean():,.0f}")
    col3.metric("Avg Frequency", f"{feats['frequency'].mean():.1f}")
    col4.metric("Avg Recency (days)", f"{feats['recency_days'].mean():.1f}")

    st.subheader("RFM Scatter (Frequency vs Monetary)")
    st.caption("Use box select / lasso to explore.")
    st.plotly_chart(px.scatter(feats, x="frequency", y="monetary", size="avg_basket_value",
                               hover_data=["customer_id","city","loyalty_member","recency_days"]), use_container_width=True)

    st.subheader("Category Share (Top categories overall)")
    with get_conn() as conn:
        tx = pd.read_sql_query("SELECT category, SUM(net_amount) as spend FROM transactions GROUP BY category ORDER BY spend DESC", conn)
    st.bar_chart(tx.set_index("category"))

    st.subheader("Heatmap: Category vs Segment (if model trained)")
    try:
        km, scaler = load_segment_model()
        if km is None:
            st.info("Train segments first to see category vs segment heatmap.")
        else:
            feats = compute_features()
            used_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
                        [c for c in feats.columns if c.startswith('share_')]
            labeled = predict_segments(feats, used_cols)
            labeled, summary = label_segments(labeled)
            pivot = labeled.merge(pd.read_sql_query("SELECT customer_id, city FROM customers", get_conn()), on='customer_id', how='left')
            # For each customer we need category spend per customer -> compute from transactions
            tx = pd.read_sql_query("SELECT customer_id, category, net_amount FROM transactions", get_conn())
            cat_cust = tx.groupby(['customer_id','category'])['net_amount'].sum().reset_index()
            merged = labeled.merge(cat_cust, on='customer_id', how='left').fillna(0)
            heat = merged.pivot_table(index='category', columns='segment_name', values='net_amount', aggfunc='sum', fill_value=0)
            if heat.empty:
                st.info("No transactions to build heatmap.")
            else:
                st.plotly_chart(px.imshow(heat, labels=dict(x="Segment", y="Category", color="Spend"),
                                          x=heat.columns, y=heat.index), use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")

# -----------------------------
# Page: Train / Retrain Models
# -----------------------------
elif page == "🧠 Train / Retrain Models":
    feats = compute_features()
    st.write("### Segment model (KMeans) training")
    st.dataframe(feats.head(10), use_container_width=True)
    k = st.slider("Choose number of segments (k)", 3, 8, 4)
    if st.button("Train / Retrain Segment Model"):
        df_tr, sil, used_cols = train_segments(feats, k=k)
        df_labeled, summary = label_segments(df_tr)
        st.success(f"Segment model trained with k={k} | Silhouette: {sil:.3f}")
        st.dataframe(summary, use_container_width=True)
        st.dataframe(df_labeled[['customer_id','segment','segment_name','recency_days','frequency','monetary']].head(20), use_container_width=True)

    st.write("### Churn model training (RandomForest)")
    churn_days = st.number_input("Churn threshold (days without purchase)", min_value=30, max_value=365, value=90)
    if st.button("Train / Retrain Churn Model"):
        feats = compute_features()
        clf, scaler, score, use_cols = train_churn_model(feats)
        st.success(f"Churn model trained. Test accuracy: {score:.3f}")

# -----------------------------
# Page: Segment Explorer
# -----------------------------
elif page == "🔎 Segment Explorer":
    feats = compute_features()
    km, scaler = load_segment_model()
    if km is None:
        st.warning("Train the segment model first from 'Train / Retrain Models'.")
    else:
        use_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
                   [c for c in feats.columns if c.startswith('share_')]
        labeled = predict_segments(feats, use_cols)
        labeled, summary = label_segments(labeled)
        seg = st.selectbox("Select Segment", sorted(labeled['segment_name'].unique().tolist()))
        view = labeled[labeled['segment_name']==seg]
        st.write(f"### Segment: {seg} (N = {len(view)})")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Customers", f"{len(view):,}")
        c2.metric("Avg Spend (₹)", f"{view['monetary'].mean():,.0f}")
        c3.metric("Avg Orders", f"{view['frequency'].mean():.1f}")
        c4.metric("Avg Recency (days)", f"{view['recency_days'].mean():.1f}")
        c5.metric("Online Ratio", f"{view['online_ratio'].mean():.2f}")
        st.write("Top Cities")
        st.bar_chart(view['city'].value_counts())
        st.write("Sample customers")
        st.dataframe(view[['customer_id','city','loyalty_member','age','gender','monetary','frequency','recency_days']].head(200), use_container_width=True)

# -----------------------------
# Page: Forecast & Recommendations
# -----------------------------
elif page == "🔮 Forecast & Recommendations":
    st.header("Forecast sales (per category) & product recommendations")
    with get_conn() as conn:
        cats = pd.read_sql_query("SELECT DISTINCT category FROM transactions", conn)['category'].tolist()
    category = st.selectbox("Select Category", options=sorted(cats))
    months = st.slider("Forecast months ahead", 1, 12, 3)
    if st.button("Run Forecast & Recommend"):
        fc_actual, fc_future = forecast_category(category, months_ahead=months)
        st.subheader(f"Monthly Sales (Category: {category})")
        if fc_actual is None:
            st.info("No data for this category.")
        else:
            combined = fc_actual.copy()
            combined = combined.rename(columns={'actual':'amount'})
            # show actual + forecast as lines
            fig = px.line(title=f"Actual vs Forecast — {category}")
            fig.add_scatter(x=combined['period'], y=combined['amount'], mode='lines+markers', name='Actual')
            fig.add_scatter(x=fc_future['period'], y=fc_future['forecast'], mode='lines+markers', name='Forecast')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended Products (Top + Trending)")
        recs = recommend_products(category, top_n=6)
        if not recs:
            st.info("No products to recommend.")
        else:
            dfrec = pd.DataFrame(recs)
            dfrec['total'] = dfrec['total'].round(2)
            dfrec['recent'] = dfrec['recent'].round(2)
            st.dataframe(dfrec[['product','total','recent','trend_score']].rename(columns={'product':'Product','total':'Total Sales','recent':'Recent Sales','trend_score':'Trend Score'}))

# -----------------------------
# Page: Churn Prediction
# -----------------------------
elif page == "💡 Churn Prediction":
    st.header("Customer Churn Prediction")
    feats = compute_features()
    if feats.empty:
        st.info("No features available.")
    else:
        st.write("Train a churn model from 'Train / Retrain Models' first if not trained.")
        clf, scaler = load_churn_model()
        if clf is None:
            st.warning("Churn model not trained.")
        else:
            st.success("Churn model loaded.")
            cust = st.text_input("Enter Customer ID to predict churn (e.g., C100042)")
            if st.button("Predict Churn for Customer"):
                row = feats[feats['customer_id']==cust]
                if row.empty:
                    st.error("Customer not found or has no transactions.")
                else:
                    use_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
                               [c for c in row.columns if c.startswith('share_')]
                    pred, proba = predict_churn_for_row(row.iloc[0], use_cols)
                    st.write(f"Predicted churn: {'Yes' if pred==1 else 'No'} — churn probability: {proba:.2f}")
            st.write("Churn-labeled sample (based on threshold 90 days)")
            sample = create_churn_labels(feats).sample(10)
            st.dataframe(sample[['customer_id','recency_days','frequency','monetary','churn']])

# -----------------------------
# Page: Customer Lookup
# -----------------------------
elif page == "👤 Customer Lookup":
    feats = compute_features()
    km, scaler = load_segment_model()
    cust_id = st.text_input("Enter Customer ID (e.g., C100042)")
    if st.button("View"):
        with get_conn() as conn:
            tx = pd.read_sql_query("SELECT * FROM transactions WHERE customer_id=? ORDER BY tx_date DESC", conn, params=[cust_id], parse_dates=["tx_date"])
            cinfo = pd.read_sql_query("SELECT * FROM customers WHERE customer_id=?", conn, params=[cust_id], parse_dates=["signup_date"])
        if cinfo.empty:
            st.error("Customer not found.")
        else:
            st.write("#### Customer Profile")
            st.table(cinfo)
            st.write("#### Recent Transactions")
            st.dataframe(tx.head(50), use_container_width=True)
            if not feats.empty:
                row = feats[feats['customer_id']==cust_id]
                if not row.empty:
                    st.write("#### Customer Features (RFM)")
                    st.table(row[['recency_days','frequency','monetary','avg_basket_value','online_ratio']].T)
                    try:
                        if km is not None:
                            used_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio'] + \
                                        [c for c in feats.columns if c.startswith('share_')]
                            pred = predict_segments(row, used_cols)
                            labeled, _ = label_segments(pred)
                            sname = labeled.iloc[0]['segment_name']
                            st.success(f"Predicted Segment: **{sname}**")
                    except Exception:
                        pass

# -----------------------------
# Page: Upload & Score CSV
# -----------------------------
elif page == "📤 Upload & Score CSV":
    st.write("""
    **Expected columns** (case-insensitive) for scoring:
    - `customer_id`, `recency_days`, `frequency`, `monetary`, `avg_basket_value`, `online_ratio`, and optional `share_*` columns.
    """)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        km, scaler = load_segment_model()
        if km is None:
            st.warning("Train model first.")
        else:
            used_cols = ['recency_days','frequency','monetary','avg_basket_value','online_ratio']
            used_cols += [c for c in df.columns if c.startswith('share_')]
            for col in used_cols:
                if col not in df.columns: df[col] = 0
            scored = predict_segments(df, used_cols)
            scored, summary = label_segments(scored)
            st.success("Scored successfully!")
            st.dataframe(scored.head(50), use_container_width=True)
            csv = scored.to_csv(index=False).encode('utf-8')
            st.download_button("Download scored CSV", data=csv, file_name="scored_customers.csv", mime="text/csv")

# -----------------------------
# End
# -----------------------------
