from collections import defaultdict
from datetime import datetime, timedelta, date
from io import BytesIO, StringIO
from pathlib import Path
import os
import csv
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, g, make_response, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from database.db import get_db, init_db

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-secret-key')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'expense_model.joblib'
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
MONTHLY_MODEL_PATH = BASE_DIR / 'monthly_classifier.joblib'
MONTHLY_FEATURES_PATH = BASE_DIR / 'monthly_features.joblib'
MONTHLY_MODEL = joblib.load(MONTHLY_MODEL_PATH) if MONTHLY_MODEL_PATH.exists() else None
MONTHLY_FEATURES = joblib.load(MONTHLY_FEATURES_PATH) if MONTHLY_FEATURES_PATH.exists() else []
init_db()

CATEGORIES = [
    'Food', 'Transport', 'Entertainment', 'Health', 'Rent', 'Shopping',
    'Utilities', 'Education', 'Travel', 'Other'
]
PAYMENT_MODES = ['UPI', 'Card', 'Cash', 'Bank Transfer', 'Other']
CATEGORY_COLORS = {
    'Food': '#c17f24',
    'Transport': '#5b7fa6',
    'Entertainment': '#8b5e83',
    'Health': '#2f7d5b',
    'Rent': '#1a472a',
    'Shopping': '#ad5f33',
    'Utilities': '#54656f',
    'Education': '#7b6aa6',
    'Travel': '#8e3b46',
    'Other': '#7f7f7f',
}
APP_NAME = 'Expense IQ'

COLUMN_ALIASES = {
    'date': ['txn date', 'transaction date', 'date', 'posting date', 'value date', 'transaction date/time'],
    'description': ['description', 'narration', 'remarks', 'transaction remarks', 'details', 'particulars', 'transaction details', 'transaction description'],
    'debit': ['dr amount', 'debit', 'debit amount', 'withdrawal', 'withdrawal amount', 'paid out', 'money out', 'debit amt'],
    'credit': ['cr amount', 'credit', 'credit amount', 'deposit', 'deposit amount', 'paid in', 'money in', 'credit amt'],
    'amount': ['amount', 'transaction amount', 'txn amount'],
    'type': ['dr/cr', 'type', 'transaction type', 'txn type', 'cr/dr'],
}

CATEGORY_KEYWORDS = {
    'Food': ['swiggy', 'zomato', 'zepto', 'blinkit', 'restaurant', 'cafe', 'dominos', 'pizza', 'burger', 'food'],
    'Transport': ['uber', 'ola', 'rapido', 'metro', 'fuel', 'petrol', 'diesel', 'irctc', 'bus', 'auto', 'transport', 'fastag'],
    'Shopping': ['amazon', 'flipkart', 'myntra', 'ajio', 'shopping', 'store', 'mart', 'retail', 'meesho'],
    'Utilities': ['airtel', 'jio', 'vi', 'electricity', 'water', 'bill', 'broadband', 'wifi', 'recharge', 'utility', 'bses'],
    'Entertainment': ['netflix', 'spotify', 'prime', 'hotstar', 'bookmyshow', 'movie', 'cinema', 'game'],
    'Health': ['pharmacy', 'hospital', 'clinic', 'apollo', 'medicine', 'lab', 'health'],
    'Rent': ['rent', 'house rent', 'flat rent', 'pg rent', 'hostel rent'],
    'Education': ['udemy', 'coursera', 'school', 'college', 'university', 'tuition', 'fees', 'course', 'book', 'library', 'exam'],
    'Travel': ['hotel', 'airbnb', 'oyo', 'makemytrip', 'goibibo', 'flight', 'airline', 'indigo', 'spicejet', 'holiday', 'trip', 'tour'],
}


@app.before_request
def load_logged_in_user():
    g.user = None
    user_id = session.get('user_id')
    if user_id:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM users WHERE id = %s', (user_id,))
                g.user = cur.fetchone()


@app.context_processor
def inject_globals():
    return {
        'current_user': g.user,
        'category_colors': CATEGORY_COLORS,
        'app_name': APP_NAME,
        'today_iso': date.today().isoformat(),
    }


@app.route('/')
def landing():
    if g.user:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        error = None
        if not name:
            error = 'Please enter your name.'
        elif not email:
            error = 'Please enter your email address.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters long.'

        if error is None:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT id FROM users WHERE email = %s', (email,))
                    existing = cur.fetchone()

                    if existing:
                        error = 'An account with this email already exists.'
                    else:
                        cur.execute(
                            'INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)',
                            (name, email, generate_password_hash(password))
                        )
                        conn.commit()
                        flash('Account created. You can sign in now.', 'success')
                        return redirect(url_for('login'))

        return render_template('register.html', error=error)

    return render_template('register.html', error=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM users WHERE email = %s', (email,))
                user = cur.fetchone()

        if user is None or not check_password_hash(user['password_hash'], password):
            return render_template('login.html', error='Invalid email or password.')

        session.clear()
        session['user_id'] = user['id']
        flash('Welcome back.', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('landing'))


def login_required():
    if g.user is None:
        flash('Please sign in to continue.', 'error')
        return redirect(url_for('login'))
    return None


def fetch_expenses(user_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT id, expense_date, description, category, payment_mode, amount
                FROM expenses
                WHERE user_id = %s
                ORDER BY expense_date DESC, id DESC
                ''',
                (user_id,)
            )
            return cur.fetchall()


def normalize_expenses(rows):
    expenses = []
    for row in rows:
        raw_date = row['expense_date']

        if isinstance(raw_date, str):
            expense_date = datetime.strptime(raw_date, '%Y-%m-%d').date()
        else:
            expense_date = raw_date

        expenses.append({
            'id': row['id'],
            'date': expense_date,
            'date_iso': expense_date.isoformat(),
            'date_display': expense_date.strftime('%d %b %Y'),
            'month_label': expense_date.strftime('%B %Y'),
            'description': row['description'],
            'category': row['category'],
            'payment_mode': row['payment_mode'],
            'amount': float(row['amount']),
        })
    return expenses


def build_grouped_views(expenses):
    by_month = defaultdict(lambda: {'total': 0.0, 'count': 0})
    by_day = defaultdict(lambda: {'total': 0.0, 'count': 0})

    for item in expenses:
        month_key = item['date'].strftime('%Y-%m')
        month_label = item['date'].strftime('%B %Y')
        by_month[month_key]['label'] = month_label
        by_month[month_key]['total'] += item['amount']
        by_month[month_key]['count'] += 1

        day_key = item['date'].isoformat()
        by_day[day_key]['label'] = item['date'].strftime('%d %b %Y')
        by_day[day_key]['total'] += item['amount']
        by_day[day_key]['count'] += 1

    month_rows = [
        {'key': key, **value}
        for key, value in sorted(by_month.items(), reverse=True)
    ]
    day_rows = [
        {'key': key, **value}
        for key, value in sorted(by_day.items(), reverse=True)
    ]
    return month_rows, day_rows


def build_daily_totals(expenses):
    daily_totals = defaultdict(float)
    for item in expenses:
        daily_totals[item['date']] += item['amount']
    return daily_totals


MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


def get_year_month_options(expenses):
    year_options = sorted({e['date'].year for e in expenses}, reverse=True)
    month_options = [{"value": month, "label": MONTH_NAMES[month]} for month in range(1, 13)]
    return year_options, month_options


def filter_expenses(expenses, *, year=None, month=None, day_iso=None, category=None):
    filtered = expenses
    if day_iso:
        filtered = [e for e in filtered if e['date_iso'] == day_iso]
    else:
        if year is not None:
            filtered = [e for e in filtered if e['date'].year == int(year)]
        if month is not None:
            filtered = [e for e in filtered if e['date'].month == int(month)]
    if category:
        filtered = [e for e in filtered if e['category'] == category]
    return filtered


def sum_expenses(expenses):
    return sum(e['amount'] for e in expenses)


def paginate_items(items, page, per_page=20):
    total_items = len(items)
    total_pages = max(1, (total_items + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return {
        'items': items[start:end],
        'page': page,
        'per_page': per_page,
        'total_items': total_items,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1,
        'next_page': page + 1,
        'start_index': start + 1 if total_items else 0,
        'end_index': min(end, total_items),
    }


def clean_col_name(name):
    value = str(name).strip().lower()
    value = re.sub(r"[\n\r\t]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def pick_best_column(columns, alias_list):
    for alias in alias_list:
        if alias in columns:
            return alias
    return None


def normalize_amount_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = (
        s.str.replace(',', '', regex=False)
         .str.replace('₹', '', regex=False)
         .str.replace('INR', '', regex=False)
         .str.replace('CR.', '', regex=False)
         .str.replace('DR.', '', regex=False)
         .str.replace('Cr.', '', regex=False)
         .str.replace('Dr.', '', regex=False)
         .str.replace('Cr', '', regex=False)
         .str.replace('Dr', '', regex=False)
         .str.strip()
    )
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, '-': pd.NA})
    return pd.to_numeric(s, errors='coerce')


def parse_statement_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    d1 = pd.to_datetime(s, errors='coerce', dayfirst=True)
    d2 = pd.to_datetime(s, errors='coerce', dayfirst=False)
    return d1 if d1.notna().sum() >= d2.notna().sum() else d2


def detect_payment_mode(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in ['upi', 'gpay', 'google pay', 'phonepe', 'paytm', 'bhim', 'imps', 'upi/']):
        return 'UPI'
    if any(k in t for k in ['atm', 'cash', 'cash withdrawal']):
        return 'Cash'
    if any(k in t for k in ['card', 'pos', 'visa', 'mastercard', 'debit card', 'credit card']):
        return 'Card'
    if any(k in t for k in ['neft', 'rtgs', 'bank transfer', 'transfer']):
        return 'Bank Transfer'
    return 'Other'


def detect_category(desc: str) -> str:
    d = str(desc).lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in d for k in keys):
            return cat
    return 'Other'


def sniff_csv_delimiter(text_sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=',;\t|')
        return dialect.delimiter
    except Exception:
        counts = {sep: text_sample.count(sep) for sep in [',', ';', '\t', '|']}
        return max(counts, key=counts.get) if any(counts.values()) else ','


def load_csv_file(raw_bytes: bytes) -> pd.DataFrame:
    decoded = None
    for encoding in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except Exception:
            continue
    if decoded is None:
        decoded = raw_bytes.decode('utf-8', errors='ignore')

    sample = decoded[:4096]
    delimiter = sniff_csv_delimiter(sample)
    df = pd.read_csv(StringIO(decoded), sep=delimiter, dtype=str, engine='python', header=None, on_bad_lines='skip')

    if df.shape[1] == 1:
        for alt in [';', ',', '\t', '|']:
            if alt == delimiter:
                continue
            alt_df = pd.read_csv(StringIO(decoded), sep=alt, dtype=str, engine='python', header=None, on_bad_lines='skip')
            if alt_df.shape[1] > df.shape[1]:
                df = alt_df
    return df


def load_excel_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    engine = 'xlrd' if filename.lower().endswith('.xls') else 'openpyxl'
    return pd.read_excel(BytesIO(raw_bytes), header=None, dtype=str, engine=engine)


def pdf_rows_from_text(page_text: str):
    rows = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'\s{2,}|\t|(?<=\d) (?=[A-Z])', line)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) >= 3:
            rows.append(parts)
    return rows


def load_pdf_file(raw_bytes: bytes) -> pd.DataFrame:
    if pdfplumber is None:
        raise ValueError('PDF import needs pdfplumber installed.')

    collected_rows = []
    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if row and any(str(cell).strip() for cell in row if cell is not None):
                        collected_rows.append([str(cell).strip() if cell is not None else '' for cell in row])
            if not tables:
                text = page.extract_text() or ''
                collected_rows.extend(pdf_rows_from_text(text))

    if not collected_rows:
        raise ValueError('No readable table data found in the PDF.')

    width = max(len(row) for row in collected_rows)
    normalized = [row + [''] * (width - len(row)) for row in collected_rows]
    return pd.DataFrame(normalized)


def read_uploaded_statement(file_storage):
    filename = (file_storage.filename or '').lower()
    raw_bytes = file_storage.read()
    file_storage.stream.seek(0)

    if filename.endswith('.csv'):
        raw = load_csv_file(raw_bytes)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        raw = load_excel_file(raw_bytes, filename)
    elif filename.endswith('.pdf'):
        raw = load_pdf_file(raw_bytes)
    else:
        raise ValueError('Unsupported file type. Upload CSV, XLSX, XLS, or PDF.')

    best_row = None
    best_score = -1
    for i, row in raw.iterrows():
        vals = [clean_col_name(x) for x in row.tolist()]
        score = 0
        for v in vals:
            if v in COLUMN_ALIASES['date']:
                score += 3
            if v in COLUMN_ALIASES['description']:
                score += 3
            if v in COLUMN_ALIASES['debit']:
                score += 2
            if v in COLUMN_ALIASES['credit']:
                score += 2
            if v in COLUMN_ALIASES['amount']:
                score += 1
            if v in COLUMN_ALIASES['type']:
                score += 1
        if score > best_score:
            best_score = score
            best_row = i

    if best_row is None or best_score < 4:
        raise ValueError('Could not detect the transaction table automatically in this file.')

    header = [clean_col_name(c) for c in raw.iloc[best_row].tolist()]
    df = raw.iloc[best_row + 1:].copy()
    df.columns = header
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def normalize_statement(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    cols = set(df.columns)

    date_col = pick_best_column(cols, COLUMN_ALIASES['date'])
    desc_col = pick_best_column(cols, COLUMN_ALIASES['description'])
    debit_col = pick_best_column(cols, COLUMN_ALIASES['debit'])
    credit_col = pick_best_column(cols, COLUMN_ALIASES['credit'])
    amount_col = pick_best_column(cols, COLUMN_ALIASES['amount'])
    type_col = pick_best_column(cols, COLUMN_ALIASES['type'])

    if date_col is None or desc_col is None:
        raise ValueError('Could not find date and description columns in this file.')

    dates = parse_statement_dates(df[date_col])
    descriptions = df[desc_col].fillna('').astype(str).str.strip()
    debit = normalize_amount_series(df[debit_col]) if debit_col else pd.Series(np.nan, index=df.index, dtype='float64')
    credit = normalize_amount_series(df[credit_col]) if credit_col else pd.Series(np.nan, index=df.index, dtype='float64')
    amount = normalize_amount_series(df[amount_col]) if amount_col else pd.Series(np.nan, index=df.index, dtype='float64')
    txn_type = df[type_col].fillna('').astype(str).str.lower().str.strip() if type_col else pd.Series('', index=df.index)

    expense_amount = pd.Series(np.nan, index=df.index, dtype='float64')

    if debit_col:
        expense_amount = debit.copy()

    if amount_col and type_col:
        dr_mask = txn_type.str.contains(r'\bdr\b|debit|withdraw|out', regex=True, na=False)
        expense_amount = pd.Series(pd.Series(amount).where(dr_mask, expense_amount), index=df.index, dtype='float64')

    if amount_col and expense_amount.isna().all():
        negative_mask = amount < 0
        expense_amount = pd.Series(amount.abs().where(negative_mask), index=df.index, dtype='float64')

    if debit_col and amount_col:
        expense_amount = debit.fillna(expense_amount)

    if credit_col:
        expense_amount = expense_amount.where(credit.isna() | (credit <= 0), pd.NA)

    out = pd.DataFrame({
        'Date': dates,
        'Description': descriptions,
        'Amount': pd.to_numeric(expense_amount, errors='coerce'),
    })
    out = out.dropna(subset=['Date', 'Amount'])
    out = out[out['Amount'] > 0]
    out = out[out['Description'].str.strip() != '']
    out = out[~out['Description'].str.lower().isin(['nan', 'none'])]
    return out.reset_index(drop=True)


def statement_to_expenses(df_norm: pd.DataFrame):
    rows = []
    for _, record in df_norm.iterrows():
        desc = str(record['Description']).strip() or '—'
        rows.append({
            'expense_date': pd.Timestamp(record['Date']).date().isoformat(),
            'description': desc,
            'category': detect_category(desc),
            'payment_mode': detect_payment_mode(desc),
            'amount': round(float(record['Amount']), 2),
        })
    return rows


def model_predict_direction(base_day, amount):
    if MODEL is None:
        return None, None

    features = pd.DataFrame([{
        'Year': base_day.year,
        'Month': base_day.month,
        'Day': base_day.day,
        'Amount': float(amount),
    }])
    pred = int(MODEL.predict(features)[0])
    probability = None
    if hasattr(MODEL, 'predict_proba'):
        probs = MODEL.predict_proba(features)[0]
        probability = float(probs[pred])
    return pred, probability


def direction_to_text(pred):
    return 'More' if pred == 1 else 'Less'


def calculate_change_pct(current_value, previous_value):
    if previous_value is None or previous_value == 0:
        return None
    return ((float(current_value) - float(previous_value)) / float(previous_value)) * 100


def classify_trend(change_pct, flat_threshold=5.0):
    if change_pct is None:
        return 'Stable'
    if change_pct >= flat_threshold:
        return 'Rising'
    if change_pct <= -flat_threshold:
        return 'Falling'
    return 'Stable'


def build_monthly_feature_frame(expenses):
    if not expenses:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            'Date': item['date'],
            'Amount': float(item['amount']),
        }
        for item in expenses
    ])
    df['Date'] = pd.to_datetime(df['Date'])
    df['MonthStart'] = df['Date'].dt.to_period('M').dt.to_timestamp()

    monthly = (
        df.groupby('MonthStart')
          .agg(Total_Amount=('Amount', 'sum'), Transaction_Count=('Amount', 'size'))
          .sort_index()
          .reset_index()
    )

    monthly['Prev_Month_Total'] = monthly['Total_Amount'].shift(1)
    monthly['Last_3_Month_Avg'] = monthly['Total_Amount'].rolling(window=3, min_periods=1).mean()
    monthly['Growth_Rate'] = np.where(
        monthly['Prev_Month_Total'].fillna(0) > 0,
        (monthly['Total_Amount'] - monthly['Prev_Month_Total']) / monthly['Prev_Month_Total'],
        0.0,
    )
    monthly['Last_Month_Change'] = monthly['Total_Amount'] - monthly['Prev_Month_Total'].fillna(monthly['Total_Amount'])
    monthly['MonthLabel'] = monthly['MonthStart'].dt.strftime('%B %Y')
    return monthly


def model_predict_month(expenses):
    if MONTHLY_MODEL is None:
        return None

    monthly = build_monthly_feature_frame(expenses)
    if monthly.empty:
        return None

    latest = monthly.iloc[-1]
    previous_total = float(monthly.iloc[-2]['Total_Amount']) if len(monthly) >= 2 else None
    recent_avg = float(monthly.tail(3)['Total_Amount'].mean()) if len(monthly) >= 1 else None
    month_change_pct = calculate_change_pct(float(latest['Total_Amount']), previous_total)
    trend = classify_trend(month_change_pct, flat_threshold=7.5)

    feature_names = list(MONTHLY_FEATURES) if isinstance(MONTHLY_FEATURES, (list, tuple)) and MONTHLY_FEATURES else []
    model_feature_names = list(getattr(MONTHLY_MODEL, 'feature_names_in_', []))
    final_features = model_feature_names or feature_names
    if not final_features:
        final_features = ['Total_Amount', 'Transaction_Count', 'Prev_Month_Total', 'Last_3_Month_Avg', 'Growth_Rate', 'Last_Month_Change']

    row = {
        'Total_Amount': float(latest['Total_Amount']),
        'Transaction_Count': int(latest['Transaction_Count']),
        'Prev_Month_Total': float(latest['Prev_Month_Total']) if pd.notna(latest['Prev_Month_Total']) else 0.0,
        'Last_3_Month_Avg': float(latest['Last_3_Month_Avg']) if pd.notna(latest['Last_3_Month_Avg']) else float(latest['Total_Amount']),
        'Growth_Rate': float(latest['Growth_Rate']) if pd.notna(latest['Growth_Rate']) else 0.0,
        'Last_Month_Change': float(latest['Last_Month_Change']) if pd.notna(latest['Last_Month_Change']) else 0.0,
    }
    features = pd.DataFrame([{name: row.get(name, 0.0) for name in final_features}])
    pred = int(MONTHLY_MODEL.predict(features)[0])
    probability = None
    if hasattr(MONTHLY_MODEL, 'predict_proba'):
        probs = MONTHLY_MODEL.predict_proba(features)[0]
        probability = float(probs[pred])

    next_month_start = (pd.Timestamp(latest['MonthStart']) + pd.offsets.MonthBegin(1)).date()
    return {
        'pred': pred,
        'probability': probability,
        'target_date': next_month_start,
        'reference_total': float(latest['Total_Amount']),
        'previous_total': previous_total,
        'recent_avg': recent_avg,
        'trend': trend,
        'trend_change_pct': month_change_pct,
        'transactions': int(latest['Transaction_Count']),
        'source_month_label': latest['MonthLabel'],
        'months_available': int(len(monthly)),
    }


def summarize_prediction_horizons(expenses, daily_totals):
    if MODEL is None and MONTHLY_MODEL is None:
        return {
            'ready': False,
            'message': 'Prediction model files are missing.',
            'latest_day': None,
            'latest_amount': None,
            'day': None,
            'month': None,
        }
    if not daily_totals:
        return {
            'ready': False,
            'message': 'Add some expenses first to unlock prediction.',
            'latest_day': None,
            'latest_amount': None,
            'day': None,
            'month': None,
        }

    latest_day = max(daily_totals.keys())
    latest_amount = float(daily_totals[latest_day])

    def bundle(label, pred, probability=None, target_date=None, compare_total=None, previous_total=None, trend=None, note=None, extra=None):
        direction = direction_to_text(pred) if pred is not None else 'Unavailable'
        tone = 'up' if pred == 1 else 'down' if pred == 0 else 'neutral'
        change_pct = calculate_change_pct(compare_total, previous_total) if compare_total is not None else None
        payload = {
            'label': label,
            'direction': direction,
            'tone': tone,
            'probability': probability,
            'change_pct': change_pct,
            'target_date': target_date.strftime('%d %b %Y') if target_date else None,
            'compare_total': compare_total,
            'previous_total': previous_total,
            'trend': trend,
            'note': note,
        }
        if extra:
            payload.update(extra)
        return payload

    day_block = None
    if MODEL is not None:
        tomorrow_pred, tomorrow_probability = model_predict_direction(latest_day, latest_amount)
        recent_dates = sorted(daily_totals.keys())
        last_7_days = recent_dates[-7:]
        prev_7_days = recent_dates[-14:-7]
        last_7_avg = sum(daily_totals[d] for d in last_7_days) / len(last_7_days) if last_7_days else None
        prev_7_avg = sum(daily_totals[d] for d in prev_7_days) / len(prev_7_days) if prev_7_days else None
        day_trend_change = calculate_change_pct(last_7_avg, prev_7_avg)
        day_block = bundle(
            'Next Day',
            tomorrow_pred,
            probability=tomorrow_probability,
            target_date=latest_day + timedelta(days=1),
            compare_total=last_7_avg,
            previous_total=prev_7_avg,
            trend=classify_trend(day_trend_change),
            note='Trend compares your recent 7-day average with the previous 7-day average.',
            extra={
                'reference_amount': latest_amount,
                'trend_change_pct': day_trend_change,
                'recent_average': last_7_avg,
            }
        )

    month_block = None
    if MONTHLY_MODEL is not None:
        month_result = model_predict_month(expenses)
        if month_result is not None:
            month_block = bundle(
                'Next Month',
                month_result['pred'],
                probability=month_result['probability'],
                target_date=month_result['target_date'],
                compare_total=month_result['reference_total'],
                previous_total=month_result['previous_total'],
                trend=month_result['trend'],
                note=f"Monthly model uses your latest full month summary from {month_result['source_month_label']}.",
                extra={
                    'trend_change_pct': month_result['trend_change_pct'],
                    'recent_average': month_result['recent_avg'],
                    'transactions': month_result['transactions'],
                    'months_available': month_result['months_available'],
                    'source_month_label': month_result['source_month_label'],
                }
            )

    ready = any(block is not None for block in [day_block, month_block])
    return {
        'ready': ready,
        'message': None if ready else 'Not enough data to generate predictions yet.',
        'latest_day': latest_day.strftime('%d %b %Y'),
        'latest_amount': latest_amount,
        'day': day_block,
        'month': month_block,
    }


def build_category_panel_data(expenses, *, default_to_current=True):
    today = datetime.now().date()
    year_options, month_options = get_year_month_options(expenses)

    if 'cat_year' in request.args:
        category_year_filter = request.args.get('cat_year', '').strip()
    elif default_to_current:
        category_year_filter = str(today.year)
    else:
        category_year_filter = ''

    if 'cat_month' in request.args:
        category_month_filter = request.args.get('cat_month', '').strip()
    elif default_to_current:
        category_month_filter = str(today.month)
    else:
        category_month_filter = ''

    category_year = int(category_year_filter) if category_year_filter.isdigit() else None
    category_month = int(category_month_filter) if category_month_filter.isdigit() else None
    category_expenses = filter_expenses(expenses, year=category_year, month=category_month)

    category_totals = {category: 0.0 for category in CATEGORIES}
    for item in category_expenses:
        category_name = item['category']
        if category_name not in category_totals:
            category_totals[category_name] = 0.0
        category_totals[category_name] += item['amount']

    category_total_amount = sum_expenses(category_expenses)
    category_cards = []
    for category, amount in sorted(category_totals.items(), key=lambda x: (-x[1], x[0])):
        share = (amount / category_total_amount * 100) if category_total_amount > 0 else 0
        category_cards.append({
            'category': category,
            'amount': amount,
            'share': share,
            'color': CATEGORY_COLORS.get(category, '#7f7f7f'),
            'detail_url': url_for(
                'category_detail',
                category=category,
                year=category_year_filter or None,
                month=category_month_filter or None,
            )
        })

    category_scope_label = 'Overall'
    if category_year and category_month:
        category_scope_label = f"{MONTH_NAMES[category_month]} {category_year}"
    elif category_year:
        category_scope_label = str(category_year)
    elif category_month:
        category_scope_label = MONTH_NAMES[category_month]

    return {
        'category_cards': category_cards,
        'category_total_amount': category_total_amount,
        'category_year_total': sum_expenses(filter_expenses(expenses, year=category_year)) if category_year else None,
        'category_scope_label': category_scope_label,
        'selected_category_year': category_year_filter,
        'selected_category_month': category_month_filter,
        'year_options': [str(y) for y in year_options],
        'month_options': month_options,
        'default_category_year': str(today.year),
        'default_category_month': str(today.month),
        'overall_total': sum_expenses(expenses),
    }


def build_dashboard_data(user_id):
    rows = fetch_expenses(user_id)
    expenses = normalize_expenses(rows)
    today = datetime.now().date()

    total = sum_expenses(expenses)
    year_expenses = filter_expenses(expenses, year=today.year)
    month_expenses = filter_expenses(expenses, year=today.year, month=today.month)
    day_expenses = filter_expenses(expenses, day_iso=today.isoformat())

    year_total = sum_expenses(year_expenses)
    month_total = sum_expenses(month_expenses)
    day_total = sum_expenses(day_expenses)

    category_panel = build_category_panel_data(expenses, default_to_current=True)

    month_rows, day_rows = build_grouped_views(expenses)
    daily_totals = build_daily_totals(expenses)
    predictions = summarize_prediction_horizons(expenses, daily_totals)
    recent_expenses = expenses[:8]

    date_range = None
    if expenses:
        oldest = min(e['date'] for e in expenses)
        newest = max(e['date'] for e in expenses)
        date_range = f"{oldest.strftime('%d-%m-%Y')} to {newest.strftime('%d-%m-%Y')}"

    return {
        'expenses': expenses,
        'recent_expenses': recent_expenses,
        'total': total,
        'count': len(expenses),
        'year_total': year_total,
        'month_total': month_total,
        'day_total': day_total,
        'current_year_label': str(today.year),
        'current_month_label': today.strftime('%B'),
        'current_day_label': today.strftime('%A'),
        'date_range': date_range,
        **category_panel,
        'month_rows': month_rows,
        'day_rows': day_rows,
        'predictions': predictions,
        'today_iso_value': today.isoformat(),
    }


@app.route('/dashboard/category-panel')
def dashboard_category_panel():
    guard = login_required()
    if guard:
        return guard
    rows = fetch_expenses(g.user['id'])
    expenses = normalize_expenses(rows)
    data = build_category_panel_data(expenses, default_to_current=True)
    response = make_response(render_template('partials/category_panel.html', **data))
    response.headers['HX-Push-Url'] = url_for(
        'dashboard',
        cat_year=data['selected_category_year'] or None,
        cat_month=data['selected_category_month'] or None,
    )
    return response


@app.route('/dashboard')
def dashboard():
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    return render_template('dashboard.html', categories=CATEGORIES, payment_modes=PAYMENT_MODES, **data)


@app.route('/expenses')
def expenses_list():
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    expenses = data['expenses']

    year_filter = request.args.get('year', '').strip()
    month_filter = request.args.get('month', '').strip()
    day_filter = request.args.get('day', '').strip()
    page = request.args.get('page', '1').strip()

    filtered = expenses
    if day_filter:
        filtered = [e for e in filtered if e['date_iso'] == day_filter]
    else:
        if year_filter.isdigit():
            filtered = [e for e in filtered if e['date'].year == int(year_filter)]
        if month_filter.isdigit():
            filtered = [e for e in filtered if e['date'].month == int(month_filter)]

    page_num = int(page) if page.isdigit() else 1
    pagination = paginate_items(filtered, page_num, per_page=20)

    return render_template(
        'expenses.html',
        expenses=pagination['items'],
        total_filtered=sum_expenses(filtered),
        selected_year=year_filter,
        selected_month=month_filter,
        selected_day=day_filter,
        year_options=sorted({str(e['date'].year) for e in expenses}, reverse=True),
        month_options=[{'value': str(month), 'label': MONTH_NAMES[month]} for month in range(1, 13)],
        pagination=pagination,
    )


@app.route('/expenses/add', methods=['GET', 'POST'])
def add_expense():
    guard = login_required()
    if guard:
        return guard

    if request.method == 'POST':
        action = request.form.get('action', 'manual').strip()

        if action == 'import_statement':
            uploaded = request.files.get('statement_file')
            if uploaded is None or not uploaded.filename:
                flash('Please choose a bank statement file first.', 'error')
                return redirect(url_for('add_expense'))

            try:
                df_raw = read_uploaded_statement(uploaded)
                df_norm = normalize_statement(df_raw)
                parsed_rows = statement_to_expenses(df_norm)
            except Exception as exc:
                flash(f'Could not import this statement: {exc}', 'error')
                return redirect(url_for('add_expense'))

            if not parsed_rows:
                flash('No debit expense rows were found in this file.', 'error')
                return redirect(url_for('add_expense'))

            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'SELECT expense_date, description, amount FROM expenses WHERE user_id = %s',
                        (g.user['id'],)
                    )
                    existing = {
                        (row['expense_date'], row['description'], round(float(row['amount']), 2))
                        for row in cur.fetchall()
                    }

                    inserted = 0
                    skipped = 0
                    for row in parsed_rows:
                        key = (row['expense_date'], row['description'], round(float(row['amount']), 2))
                        if key in existing:
                            skipped += 1
                            continue
                        cur.execute(
                            '''
                            INSERT INTO expenses (user_id, expense_date, description, category, payment_mode, amount)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ''',
                            (g.user['id'], row['expense_date'], row['description'], row['category'], row['payment_mode'], row['amount'])
                        )
                        existing.add(key)
                        inserted += 1

                conn.commit()

            if inserted:
                flash(f'Imported {inserted} expenses from the statement. Skipped {skipped} duplicates.', 'success')
            else:
                flash(f'No new expenses were added. Skipped {skipped} duplicates.', 'error')
            return redirect(url_for('expenses_list'))

        expense_date = request.form.get('expense_date', '').strip()
        description = request.form.get('description', '').strip() or '—'
        category = request.form.get('category', 'Other').strip()
        payment_mode = request.form.get('payment_mode', 'Other').strip()
        amount_raw = request.form.get('amount', '0').strip()

        try:
            amount = float(amount_raw)
        except ValueError:
            amount = 0.0

        try:
            datetime.strptime(expense_date, '%Y-%m-%d')
        except ValueError:
            flash('Please enter a valid date.', 'error')
            return redirect(url_for('add_expense'))

        if amount <= 0:
            flash('Amount must be greater than 0.', 'error')
            return redirect(url_for('add_expense'))

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO expenses (user_id, expense_date, description, category, payment_mode, amount)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ''',
                    (g.user['id'], expense_date, description, category, payment_mode, amount)
                )
            conn.commit()

        flash('Expense added successfully.', 'success')
        return redirect(url_for('expenses_list'))

    return render_template('add_expense.html', categories=CATEGORIES, payment_modes=PAYMENT_MODES)


@app.route('/details')
def details_overview():
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    return render_template(
        'details.html',
        month_rows=data['month_rows'],
    )


@app.route('/details/total')
def total_detail():
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    return render_template(
        'detail_list.html',
        page_title='All Transaction Details',
        page_subtitle='All expenses across your full record range.',
        expenses=data['expenses'],
        total=sum_expenses(data['expenses']),
        back_url=url_for('dashboard'),
    )


@app.route('/details/year/<int:year>')
def year_detail(year):
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    valid_years = {int(y) for y in data['year_options']}
    if year not in valid_years:
        flash('Invalid year selected.', 'error')
        return redirect(url_for('details_overview'))
    expenses = filter_expenses(data['expenses'], year=year)
    return render_template(
        'detail_list.html',
        page_title=f"{year} Details",
        page_subtitle='All expenses logged for this year.',
        expenses=expenses,
        total=sum_expenses(expenses),
        back_url=url_for('dashboard'),
        selector_mode='year',
        year_options=data['year_options'],
        selected_year=str(year),
    )


@app.route('/details/month/<period>')
def month_detail(period):
    guard = login_required()
    if guard:
        return guard
    try:
        target = datetime.strptime(period, '%Y-%m')
    except ValueError:
        flash('Invalid month selected.', 'error')
        return redirect(url_for('details_overview'))

    data = build_dashboard_data(g.user['id'])
    month_expenses = filter_expenses(data['expenses'], year=target.year, month=target.month)

    day_totals = defaultdict(lambda: {'total': 0.0, 'count': 0})
    for item in month_expenses:
        day_key = item['date_iso']
        day_totals[day_key]['label'] = item['date'].strftime('%d %b %Y')
        day_totals[day_key]['total'] += item['amount']
        day_totals[day_key]['count'] += 1

    day_rows = [
        {'key': key, **value}
        for key, value in sorted(day_totals.items(), reverse=True)
    ]

    return render_template(
        'month_days.html',
        month_period=period,
        page_title=f"{target.strftime('%B %Y')}",
        page_subtitle="Open a day folder to view that day's expenses.",
        days=day_rows,
        total=sum_expenses(month_expenses),
        entry_count=len(month_expenses),
        back_url=url_for('details_overview'),
    )


@app.route('/details/day/<day_iso>')
def day_detail(day_iso):
    guard = login_required()
    if guard:
        return guard
    try:
        target = datetime.strptime(day_iso, '%Y-%m-%d').date()
    except ValueError:
        flash('Invalid day selected.', 'error')
        return redirect(url_for('details_overview'))

    data = build_dashboard_data(g.user['id'])
    expenses = filter_expenses(data['expenses'], day_iso=target.isoformat())
    return render_template(
        'detail_list.html',
        page_title=f"{target.strftime('%d %b %Y')} Details",
        page_subtitle='All expenses logged for this day.',
        expenses=expenses,
        total=sum_expenses(expenses),
        back_url=url_for('month_detail', period=target.strftime('%Y-%m')),
        selector_mode='day',
        selected_day=target.isoformat(),
    )


@app.route('/details/category/<category>')
def category_detail(category):
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    year_filter = request.args.get('year', '').strip()
    month_filter = request.args.get('month', '').strip()
    year = int(year_filter) if year_filter.isdigit() else None
    month = int(month_filter) if month_filter.isdigit() else None
    expenses = filter_expenses(data['expenses'], year=year, month=month, category=category)

    scope = 'overall records'
    if year and month:
        scope = f"{MONTH_NAMES[month]} {year}"
    elif year:
        scope = f"{year}"
    elif month:
        scope = f"all {MONTH_NAMES[month]} records"

    return render_template(
        'detail_list.html',
        page_title=f"{category} Details",
        page_subtitle=f"All {category.lower()} expenses for {scope}.",
        expenses=expenses,
        total=sum_expenses(expenses),
        back_url=url_for('dashboard', cat_year=year_filter or None, cat_month=month_filter or None),
        selector_mode='category',
        year_options=data['year_options'],
        month_options=data['month_options'],
        selected_year=year_filter,
        selected_month=month_filter,
        selected_category=category,
    )


@app.route('/prediction')
def prediction_page():
    guard = login_required()
    if guard:
        return guard
    data = build_dashboard_data(g.user['id'])
    return render_template('prediction.html', predictions=data['predictions'])


@app.route('/expenses/<int:id>/edit', methods=['GET', 'POST'])
def edit_expense(id):
    guard = login_required()
    if guard:
        return guard

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT * FROM expenses WHERE id = %s AND user_id = %s',
                (id, g.user['id'])
            )
            expense = cur.fetchone()

        if expense is None:
            flash('Expense not found.', 'error')
            return redirect(url_for('expenses_list'))

        if request.method == 'POST':
            expense_date = request.form.get('expense_date', '').strip()
            description = request.form.get('description', '').strip() or '—'
            category = request.form.get('category', 'Other').strip()
            payment_mode = request.form.get('payment_mode', 'Other').strip()
            amount_raw = request.form.get('amount', '0').strip()

            try:
                amount = float(amount_raw)
            except ValueError:
                amount = 0.0

            try:
                datetime.strptime(expense_date, '%Y-%m-%d')
            except ValueError:
                flash('Please enter a valid date.', 'error')
                return redirect(url_for('edit_expense', id=id))

            if amount <= 0:
                flash('Amount must be greater than 0.', 'error')
                return redirect(url_for('edit_expense', id=id))

            with conn.cursor() as cur:
                cur.execute(
                    '''
                    UPDATE expenses
                    SET expense_date = %s, description = %s, category = %s, payment_mode = %s, amount = %s
                    WHERE id = %s AND user_id = %s
                    ''',
                    (expense_date, description, category, payment_mode, amount, id, g.user['id'])
                )
            conn.commit()
            flash('Expense updated.', 'success')
            return redirect(url_for('expenses_list'))

    return render_template('edit_expense.html', expense=expense, categories=CATEGORIES, payment_modes=PAYMENT_MODES)


@app.route('/expenses/<int:id>/delete', methods=['POST'])
def delete_expense(id):
    guard = login_required()
    if guard:
        return guard

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM expenses WHERE id = %s AND user_id = %s', (id, g.user['id']))
        conn.commit()

    flash('Expense deleted.', 'success')
    return redirect(request.referrer or url_for('expenses_list'))


@app.route('/profile')
def profile():
    guard = login_required()
    if guard:
        return guard
    return render_template('profile.html')


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
