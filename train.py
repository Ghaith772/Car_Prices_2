# لو xgboost مش متثبت عندك: pip install xgboost
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd

df = pd.read_csv("hatla2ee_cars_august_2025.csv")  # عدّل اسم الملف حسب عندك

print("عدد الصفوف الكلي:", len(df))

# ============================================================
# 1) تحويل price و mileage إلى double
# ============================================================
df['price'] = (
    df['price'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('EGP', '', regex=False)
    .str.strip()
)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['mileage'] = (
    df['mileage'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('Km', '', regex=False)
    .str.strip()
)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

# ============================================================
# 2) حذف الأسعار الوهمية (أرقام مكررة بالكامل زي 11,111,111 أو 9,999,999)
#    دي مش أسعار حقيقية، غالبًا حد داخل رقم عشوائي/تجريبي وقت الإعلان
# ============================================================
def is_repdigit_price(x):
    if pd.isnull(x):
        return False
    s = str(int(x))
    return len(s) >= 4 and len(set(s)) == 1

mask_fake_price = df['price'].apply(is_repdigit_price)
n_fake_price = mask_fake_price.sum()
df = df[~mask_fake_price]
print(f"تم حذف {n_fake_price} صف بسعر وهمي (أرقام مكررة بالكامل)  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 3) حل مشكلة تقسيم اسم البراند المكوّن من كلمتين
# ============================================================
brand_fix = {
    'Alfa': 'Alfa Romeo',
    'Great': 'Great Wall',
    'Land': 'Land Rover',
    'Ssang': 'Ssang Yong',
    'King': 'King Long',
}
mask = df['company'].isin(brand_fix.keys())

def fix_row(row):
    correct_company = brand_fix[row['company']]
    title = str(row['title']).strip()
    rest = title[len(correct_company):].strip()
    year = row['year']
    if pd.notnull(year):
        year_str = str(int(year))
        if rest.endswith(year_str):
            rest = rest[:-len(year_str)].strip()
    return pd.Series({'company': correct_company, 'model': rest})

fixed_cols = df.loc[mask].apply(fix_row, axis=1)
df.loc[mask, 'company'] = fixed_cols['company']
df.loc[mask, 'model'] = fixed_cols['model']

# ============================================================
# 4) حل مشكلة year = رقم الموديل
# ============================================================
year_str = df['year'].astype(str).str.replace('.0', '', regex=False)
mask_year = year_str == df['model'].astype(str)

def fix_year(row):
    title = str(row['title']).strip()
    last_token = title.split()[-1]
    return float(last_token)

df.loc[mask_year, 'year'] = df.loc[mask_year].apply(fix_year, axis=1)

# ============================================================
# 5) تطبيع company و model: حذف أي حرف مش رقم أو حرف إنجليزي (مسافات،
#    شرطات، فواصل...) وتوحيد الحالة (أول حرف كبير والباقي صغير)
#    مثال: "Range Rover" -> "Rangerover"
# ============================================================
for col in ['company', 'model' , 'color', 'transmission']:
    df[col] = df[col].str.replace(r'[^A-Za-z0-9]', '', regex=True).str.capitalize()
print("تم تطبيع company و model (حذف الرموز/المسافات + حرف أول كبير بس)")

# ============================================================
# 6) حذف الأعمدة: title, location, date_posted, detail_link
# ============================================================
df = df.drop(columns=['title', 'location', 'date_posted', 'detail_link'])
print("تم حذف: title, location, date_posted, detail_link")

# ============================================================
# 7) حذف الصفوف المكررة بالكامل
# ============================================================
n_dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"تم حذف {n_dupes} صف مكرر  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 8) حذف الصفوف اللي model == 'Other'
# ============================================================
n_other = (df['model'] == 'Other').sum()
df = df[df['model'] != 'Other']
print(f"تم حذف {n_other} صف model=Other  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 9) حذف الصفوف اللي price مفقود
# ============================================================
n_price_null = df['price'].isnull().sum()
df = df[df['price'].notnull()]
print(f"تم حذف {n_price_null} صف price مفقود  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 10) حذف الصفوف اللي year مفقود
# ============================================================
n_year_null = df['year'].isnull().sum()
df = df[df['year'].notnull()]
print(f"تم حذف {n_year_null} صف year مفقود  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 11) ملء القيم المفقودة في mileage بالوسيط
# ============================================================
median_mileage = df['mileage'].median()
n_mileage_null = df['mileage'].isnull().sum()
df['mileage'] = df['mileage'].fillna(median_mileage)
print(f"تم ملء {n_mileage_null} صف في mileage بالوسيط ({median_mileage} كم)")

# ============================================================
# 12) تحويل price من EGP إلى USD
# ============================================================
df['price'] = df['price'] / 50
print("تم تحويل price من EGP لـ USD (بالقسمة على 50)")

# ============================================================
# 13) حذف outliers من price (حد يدوي: أكبر من مليون دولار)
# ============================================================
n_price_outliers = (df['price'] > 1_000_000).sum()
df = df[df['price'] <= 1_000_000]
print(f"تم حذف {n_price_outliers} صف price outlier (>$1,000,000)  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 14) حذف outliers من mileage بطريقة IQR (×3)
# ============================================================
q1 = df['mileage'].quantile(0.25)
q3 = df['mileage'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 3 * iqr
n_mileage_outliers = (df['mileage'] > upper_bound).sum()
df = df[df['mileage'] <= upper_bound]
print(f"تم حذف {n_mileage_outliers} صف mileage outlier  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 15) حذف outliers من price المنخفض (حد يدوي: أقل من $200)
# ============================================================
n_low_price = (df['price'] < 200).sum()
df = df[df['price'] >= 200]
print(f"تم حذف {n_low_price} صف price منخفض جدًا (<$200)  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 16) حذف أسعار شاذة مقارنة بنظرائها (نفس company+model+year)
#    لو سعر إعلان أكبر من 5 أضعاف متوسط سعر نفس السيارة (نفس الشركة
#    والموديل والسنة)، وده مبني على 3 إعلانات مماثلة على الأقل عشان
#    نتأكد إن المقارنة فيها معنى، فده على الأرجح خطأ إدخال (زفلتة صفر)
#    مش سعر حقيقي.
# ============================================================
peer_group = ['company', 'model', 'year']
peer_size = df.groupby(peer_group)['price'].transform('size')
peer_median = df.groupby(peer_group)['price'].transform('median')
price_ratio = df['price'] / peer_median

mask_price_error = (peer_size >= 3) & (price_ratio > 5)
n_price_error = mask_price_error.sum()
df = df[~mask_price_error]
print(f"تم حذف {n_price_error} صف سعره أكبر من 5 أضعاف متوسط أشباهه (نفس الشركة/الموديل/السنة)  -> الصفوف المتبقية: {len(df)}")

# ============================================================
# 17) اشتقاق transmission من features
# ============================================================
def derive_transmission(features):
    if pd.isnull(features):
        return 'manual'
    return 'automatic' if 'automatic' in str(features).lower() else 'manual'

df['transmission'] = df['features'].apply(derive_transmission)
print("توزيع transmission بعد الاشتقاق:")
print(df['transmission'].value_counts())
for col in [ 'transmission']:
    df[col] = df[col].str.replace(r'[^A-Za-z0-9]', '', regex=True).str.capitalize()
# ============================================================
# 18) حذف عمود features (بعد ما استخدمناه فوق)
# ============================================================
df = df.drop(columns=['features'])
print("تم حذف عمود: features")

# ============================================================
# حفظ الملف النظيف
# ============================================================
df.to_csv("hatla2ee_cars_cleaned.csv", index=False)
print(f"العدد النهائي للصفوف: {len(df)}")
print("تم الحفظ في: hatla2ee_cars_cleaned.csv")

# ============================================================
# 19) تجهيز البيانات وتدريب موديل XGBoost للتنبؤ بالسعر
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

model_df = df.copy()

categorical_cols = ['company', 'model', 'color', 'transmission']
for col in categorical_cols:
    model_df[col] = model_df[col].astype('category')

# حفظ الفئات (categories) بالضبط متل ما شافها الموديل وقت التدريب.
# predict.py لازم يطبّق نفس الفئات على أي بيانات جديدة قبل التوقع،
# وإلا الموديل رح يفسّر الأرقام بشكل غلط.
saved_categories = {col: model_df[col].cat.categories.tolist() for col in categorical_cols}
with open("categories.json", "w", encoding="utf-8") as f:
    json.dump(saved_categories, f, ensure_ascii=False)
print("تم حفظ الفئات في: categories.json")

X = model_df.drop(columns=['price'])
y = model_df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = xgb_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print("نتائج تدريب موديل XGBoost للتنبؤ بسعر السيارة")
print("=" * 60)
print(f"R² (معامل التحديد): {r2:.4f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE : ${mae:,.2f}")

importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nأهم العوامل المؤثرة على السعر:")
print(importance)

xgb_model.save_model("car_price_xgb_model.json")
print("\nتم حفظ الموديل في: car_price_xgb_model.json")